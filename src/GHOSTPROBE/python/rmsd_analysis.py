import pandas as pd
import mdtraj 
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pymp
import pickle
import os
import sys

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_gro', nargs="?", help="system coordinates gro",default="system.gro")
    parser.add_argument('-t','--input_traj', nargs="+", help="trajectory file(s)",default=["traj.xtc"])
    parser.add_argument('-s','--stride', nargs="?", help="stride to load trajectory",default=1)
    parser.add_argument('-r','--rmsd', nargs="?", help="pkl file containing the rmsd matrix",default="pairwise.pkl")
    parser.add_argument('-a','--align', nargs="?", help="atoms you want to align to reference (VMD style selection)",default="backbone")
    parser.add_argument('-rho','--rho',nargs="?",help="d0 value for laio clustering",type=float,default=0.1)
    parser.add_argument('-delta','--delta',nargs="?",help="minimum delta value for a point to be considered a laio cluster center",type=float,default=None)
    parser.add_argument('-o','--output_csv', nargs="?", help="output csv file",default="rmsd.csv")
    args = parser.parse_args()
    return args

def calc_pairwise_rmsd(traj,align_set):
    traj_align_idx=traj.topology.select(align_set)
    rmsd=pymp.shared.array((traj.n_frames,traj.n_frames),dtype=np.float64)
    with pymp.Parallel(1) as p:
       for i in p.range(0,traj.n_frames):
           rmsd[i]=mdtraj.rmsd(traj,traj,frame=i,atom_indices=traj_align_idx)
           print("Finished processing frame %i" % i)
    print("Max pairwise rmsd: ", np.max(rmsd))
    pickle.dump(rmsd,open("pairwise.pkl","wb"))
    return rmsd

def laio(rmsd,rmsd_threshold,delta_min):
    rhodelta=pd.DataFrame()
    rhodelta["point"]=range(0,len(rmsd))
    rhodelta["rho"]=[0]*len(rmsd)
    rhodelta["nnhd"]=[-1]*len(rmsd)
    rhodelta["delta"]=[0.]*len(rmsd)
    rhodelta["cluster"]=[-1]*len(rmsd)
    rhodelta["cluster_centre"]=[-1]*len(rmsd)
    
    #Assign density
    for i in range(0,len(rhodelta)):
        rhodelta.rho[i]=len(np.where(rmsd[i]<=rmsd_threshold)[0])
   
    #calculate delta
    for i in range(0,len(rhodelta)):
        delta_i=np.inf
        nnhd_i=-1
        for j in range(0,len(rhodelta)):
            if (i==j):
                continue
            if (rhodelta.rho[j]>rhodelta.rho[i] and rmsd[i][j]<delta_i):
                delta_i=rmsd[i][j]
                nnhd_i=j
        if (delta_i==np.inf):
            delta_i=max(rmsd[i])
            nnhd_i=-1
        rhodelta.nnhd[i]=nnhd_i
        rhodelta.delta[i]=delta_i
    
    #choose cluster centers
    print(rhodelta.sort_values("delta",ascending=False)[0:50])
    
    if delta_min is None:
       z="start"
       while (type(z)!=float):
           z=input("Please indicate the minimum delta for a point to be considered a cluster centre\n")
           try: 
               z=float(z)
           except:
               print("Looks like you didn't give me a float. Try again\n")
    else:
        z=delta_min

    cluster_centers=[]
    for i in range(0,len(rhodelta)):
        if rhodelta.delta[i]>=z:
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
    print(rhodelta)
    rhodelta.sort_values("point",inplace=True,ignore_index=True)
    rhodelta.to_pickle("rhodelta.pkl")
    return rhodelta

def save_clusters(rhodelta,traj):
    clusters={}
    for i in range(0,len(rhodelta)):
        if rhodelta.cluster_centre[i] not in clusters.keys():
            clusters[rhodelta.cluster_centre[i]]=[rhodelta.point[i]]
        else:
            clusters[rhodelta.cluster_centre[i]].append(rhodelta.point[i])
    
    for key in clusters.keys():
        name="cluster_"+str(key)
        traj[key].save_gro(name+".gro")
        trjslice=traj[clusters[key]]
        trjslice.save_xtc(name+".xtc")

if __name__=="__main__":
   
   args=parse()
   print("The atom selection for alignment is:")
   print(args.align)
   
   traj=mdtraj.load(args.input_traj,top=args.input_gro,stride=args.stride)
   try:
     pairwise_rmsd=pickle.load(open(args.rmsd,"rb"))
     print("Loaded file %s" % args.rmsd)
   except:
     print("File %s could not be opened, will calculate pairwise RMSD" % args.rmsd) 
     pairwise_rmsd=calc_pairwise_rmsd(traj,args.align)
   rhodelta=laio(pairwise_rmsd,args.rho,args.delta)
   save_clusters(rhodelta,traj)
   


   
   
   