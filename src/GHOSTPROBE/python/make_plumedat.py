import argparse
import pandas as pd
import numpy as np
import mdtraj
import glob
import sys

#default line
defline="""GHOSTPROBE ...
LABEL=d
NPROBES=16 PROBESTRIDE=2500
KPERT=0.1 PERTSTRIDE=25000
RMIN=0 DELTARMIN=0.4
RMAX=0.45 DELTARMAX=0.30
CMIN=0 DELTAC=1
PMIN=10 DELTAP=10
"""

biasline="RESTRAINT ARG=d AT=1.0 KAPPA=500 STRIDE=10\n"
printline="PRINT ARG=d FILE=COLVAR STRIDE=2500"


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--regex', nargs="?", type=str, help="Regex to find all probe-$i-stats.csv files",default=None)
    parser.add_argument('-f','--input_gro', nargs="?", type=str, help="Reference structure (issued by gmx trjconv)",default="md_dry.gro")
    parser.add_argument('-s','--selection', nargs="?", type=str, help="Selection of atoms that will go in plumed",default="protein")
    parser.add_argument('-a','--activity',nargs="?", type=float,help="Minimum probe activity to be considered",default=1.0)
    parser.add_argument('-rho','--rho',nargs="?", type=float,help="d0 value for laio clustering",default=0.5)
    parser.add_argument('-delta','--delta',nargs="?", type=float,help="minimum delta value for a point to be considered a laio cluster center",default=0.5)
    parser.add_argument('-e','--min_elements', nargs="?", type=int, help="Min elements in a proper cluster",default=1)
    args = parser.parse_args()
    print(args)
    return args

def getATOMS(gro,selection):
    structure=mdtraj.load(gro)
    atomlist=[]
    idlist=structure.topology.select(selection)
    for id in idlist:
            atom=structure.topology.atom(id)
            if (atom.element==mdtraj.element.hydrogen):
                continue
            atomlist.append(str(atom.serial))
    ATOMS="ATOMS="+",".join(atomlist)
    return ATOMS

def calculate_distance_matrix(xyzmat):
    distmat=np.zeros((len(xyzmat),len(xyzmat)),dtype=np.float64)
    #print(type(distmat))
    for i in range(0,len(xyzmat)):
        for j in range(i,len(xyzmat)):
            if (j==i):
                continue
            else:
                distmat[i][j]=np.linalg.norm(xyzmat[i]-xyzmat[j])
                distmat[j][i]=distmat[i][j]
    return distmat


def laio(id,names,distmat,d0,delta_min):
    rhodelta=pd.DataFrame()
    rhodelta["point"]=range(0,len(distmat))
    rhodelta["rho"]=[0]*len(distmat)
    rhodelta["nnhd"]=[-1]*len(distmat)
    rhodelta["delta"]=[0.]*len(distmat)
    rhodelta["cluster"]=[-1]*len(distmat)
    rhodelta["cluster_centre"]=[-1]*len(distmat)
    rhodelta["ID"]=id
    rhodelta["Name"]=names
    
    #Assign density
    for i in range(0,len(rhodelta)):
        rhodelta.rho[i]=len(np.where(distmat[i]<=d0)[0])
   
    #calculate delta
    for i in range(0,len(rhodelta)):
        delta_i=np.inf
        nnhd_i=-1
        for j in range(0,len(rhodelta)):
            if (i==j):
                continue
            if (rhodelta.rho[j]>rhodelta.rho[i] and distmat[i][j]<delta_i):
                delta_i=distmat[i][j]
                nnhd_i=j
        if (delta_i==np.inf):
            delta_i=max(distmat[i])
            nnhd_i=-1
        rhodelta.nnhd[i]=nnhd_i
        rhodelta.delta[i]=delta_i
    
    #choose cluster centers
    #print(rhodelta.sort_values("delta",ascending=False)[0:50])
    
    

    cluster_centers=[]
    cluster_centers_id=[]
    for i in range(0,len(rhodelta)):
        if rhodelta.delta[i]>=delta_min:
            cluster_centers.append(rhodelta.point[i])

    #Assign cluster ids to cluster centres
    for i in range(0,len(cluster_centers)):
        point_i=cluster_centers[i]
        rhodelta.cluster[point_i]=i
        rhodelta.cluster_centre[point_i]=point_i

    #Assign rest of points to clusters
    rhodelta.sort_values("rho",ascending=False,inplace=True,ignore_index=True)
    for i in range(0,len(rhodelta)):
        if rhodelta.cluster[i]!=-1:
            continue
        nnhd=rhodelta.nnhd[i]
        #print(np.where(rhodelta.point==nnhd))
        #return
        nnhd_j=np.where(rhodelta.point==nnhd)[0][0] #index of rhodelta.nnhd[i] in the sorted dataframe
        rhodelta.cluster[i]=rhodelta.cluster[nnhd_j]
        rhodelta.cluster_centre[i]=rhodelta.cluster_centre[nnhd_j]
        #print(nnhd,nnhd_j,rhodelta.cluster[nnhd_j])
    
    #pickle rhodelta
    rhodelta.sort_values("cluster",inplace=True,ignore_index=True)
    rhodelta.to_csv("rhodelta.csv", sep=" ")
    return rhodelta

def find_unique_atoms(regex,activity_min):
    stats=glob.glob(regex)
    for i in range(0,len(stats)):
        if i==0:
            z=pd.read_csv(stats[i],sep=" ")
        else:
            z=z.append(pd.read_csv(stats[i],sep=" "))
    
    z=z[z.activity>=activity_min]
    atoms_id=np.sort(z.min_r_serial.unique())
    #print(atoms_id)
    return atoms_id

def get_atom_crd(gro,atoms_id):
    crd=[]
    names=[]
    grobj=mdtraj.load(gro)
    for id in atoms_id:
        mdtraj_id=id-1 # PLumed serials start at 1
        crd.append(grobj.xyz[0][mdtraj_id])
        names.append(grobj.topology.atom(mdtraj_id))
    
    crd=np.array(crd)
    #print(crd)
    return crd, names

def print_clusters(rhodelta,min_elements):
    lines=[]
    clusters=rhodelta.cluster.unique()
    rhodeltaclust_lst=[]
    for cluster_id in clusters:
        rhodeltaclust=rhodelta[rhodelta.cluster==cluster_id]
        rhodeltaclust_lst.append(rhodeltaclust)
    rhodeltaclust_lst.sort(key=len, reverse=True)
    for i in range(0,len(rhodeltaclust_lst)):
        rhodeltaclust=rhodeltaclust_lst[i]
        print(rhodeltaclust)
        print(len(rhodeltaclust.point))
        print("---------------------")
        if (len(rhodeltaclust)>=min_elements):
           filename="cluster_"+str(i)+".csv"
           rhodeltaclust.to_csv(filename,sep=" ")
           line="ATOMS_INIT="+",".join(rhodeltaclust.astype({"ID":str}).ID)
           lines.append(line)
    #sys.exit()
    return lines

def build_plumedat(defline,ATOMS,lines):
    defline=defline+ATOMS+"\n"
    if len(lines)>0:
       #lines=list(sorted(lines,key=len,reverse=True))
       for i in range(0,len(lines)):
         name="plumed_"+str(i)+".dat"
         fileout=open(name,"w")
         fileout.write(defline)
         fileout.write(lines[i]+"\n")
         fileout.write("... SPHDRUG\n")
         fileout.write(biasline)
         fileout.write(printline)
         fileout.close()
    else:
        defline=defline+"... SPHDRUG\n"
        fileout=open("plumed.dat","w")
        fileout.write(defline)
        fileout.write(biasline)
        fileout.write(printline)
        fileout.close()
    

if __name__=="__main__":

    args=parse()
    atomlist=getATOMS(args.input_gro,args.selection)
    lines=[]
    if args.regex is not None:
       atoms_id=find_unique_atoms(args.regex,args.activity)
       crd,names=get_atom_crd(args.input_gro,atoms_id)
       distmat=calculate_distance_matrix(crd)
       rhodelta=laio(atoms_id,names,distmat,args.rho,args.delta)
       lines=print_clusters(rhodelta,args.min_elements)
    
    build_plumedat(defline,atomlist,lines)
    
    

    

