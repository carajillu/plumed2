import pandas as pd
import mdtraj 
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pymp
import pickle

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_gro', nargs="?", help="system coordinates gro",default="system.gro")
    parser.add_argument('-t','--input_traj', nargs="+", help="trajectory file(s)",default=["traj.xtc"])
    parser.add_argument('-s','--stride', nargs="?", help="stride to load trajectory",default=1)
    parser.add_argument('-r','--ref', nargs="+", help="reference structure(s) gro or pdb",default=["ref.pdb"])
    parser.add_argument('-a','--align', nargs="?", help="atoms you want to align to reference (VMD style selection)",default="backbone")
    parser.add_argument('-d','--rmsd', nargs="?", help="atoms you want to align to reference (VMD style selection)",default="backbone")
    parser.add_argument('-u','--rmsd_threshold',nargs="?",help="rmsd threshold for clustering",type=float,default=0.1)
    parser.add_argument('-o','--output_csv', nargs="?", help="output csv file",default="rmsd.csv")
    args = parser.parse_args()
    return args

def get_atom_set(ref,selection,input_gro):
   #Get atoms from references (make sure they are the same)
   atoms_refs=[]
   for reference_pdb in ref:
      ref_i=mdtraj.load(reference_pdb)
      sel=ref_i.topology.select(selection)
      atoms=[]
      for atom_id in sel:
          atoms.append(str(ref_i.topology.atom(atom_id)))
      atoms_refs.append(atoms)

   #Get atoms from trajectory   
   traj=mdtraj.load(input_gro)
   sel=traj.topology.select(selection)
   atoms=[]
   for atom_id in sel:
          atoms.append(str(traj.topology.atom(atom_id)))      
   atoms_refs.append(atoms)

   align_set=set(atoms_refs[0])
   for s in atoms_refs[1:]:
       align_set.intersection_update(s)
   print(align_set)    
   return align_set

def get_atomset_idx(traj,atomset):
    idx_set=[]
    for atom in traj.topology.atoms:
        if str(atom) in atomset:
            idx_set.append(atom.index)
    return idx_set

def calc_rmsd(traj,ref, align_set, rmsd_set):
    traj_align_idx=get_atomset_idx(traj,align_set)
    traj_rmsd_idx=get_atomset_idx(traj,rmsd_set)
    rmsd=pd.DataFrame()
    for reference_pdb in ref:
        ref_i=mdtraj.load(reference_pdb)
        ref_align_idx=get_atomset_idx(ref_i,align_set)
        ref_rmsd_idx=get_atomset_idx(ref_i,rmsd_set)
        traj=traj.superpose(reference=ref_i,atom_indices=traj_align_idx,ref_atom_indices=ref_align_idx)
        #RMSD without superposition
        crd_ref=ref_i.xyz[0][ref_rmsd_idx]
        rms_ref=[]
        for frame in traj.xyz:
            crd_trj=frame[traj_rmsd_idx]
            rms=np.sqrt(((((crd_trj-crd_ref)**2))*3).mean())
            rms_ref.append(rms)
            #print(rms)
        rmsd[reference_pdb]=rms_ref
        #rmsd[reference_pdb]=mdtraj.rmsd(traj,ref_i,atom_indices=traj_rmsd_idx,ref_atom_indices=ref_rmsd_idx)
    return rmsd

def calc_pairwise_rmsd(traj,align_set, rmsd_set):
    traj_align_idx=get_atomset_idx(traj,align_set)
    traj_rmsd_idx=get_atomset_idx(traj,rmsd_set)
    #traj=traj.superpose(reference=traj[0],atom_indices=traj_align_idx)
    rmsd=pymp.shared.array((traj.n_frames,traj.n_frames),dtype=np.float64)
    with pymp.Parallel() as p:
       rmsd_i=np.zeros((traj.n_frames,traj.n_frames),dtype=np.float64)
       for i in p.range(0,traj.n_frames):
           frame_i=traj[i]
           crd_i=frame_i.xyz[0][traj_rmsd_idx]
           for j in range(i,traj.n_frames):
               frame_j=traj[j]
               frame_j=frame_j.superpose(reference=frame_i,atom_indices=traj_align_idx)
               crd_j=frame_j.xyz[0][traj_rmsd_idx]
               rmsd_i[i][j]=np.sqrt(((((crd_i-crd_j)**2))*3).mean())
               rmsd_i[j][i]=rmsd_i[i][j]
           print("Finished processing frame %i" % i)
       with p.lock: 
           rmsd+=rmsd_i
    print("Max pairwise rmsd: ", np.max(rmsd))
    pickle.dump(rmsd,open("pairwise.pkl","wb"))
    return rmsd

def laio(rmsd,rmsd_threshold):
    rhodelta=pd.DataFrame()
    rhodelta["point"]=range(0,len(rmsd))
    rhodelta["rho"]=[0]*len(rmsd)
    rhodelta["nnhd"]=[-1]*len(rmsd)
    rhodelta["delta"]=[0.]*len(rmsd)
    rhodelta["cluster"]=[-1]*len(rmsd)
    
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
    print(rhodelta.sort_values("delta",ascending=False))
    cluster_centers=[]
    z="start"
    while (z!=""):
        z=input("Please indicate a point that will be a cluster centre. If non left, press enter:\n")
        if z=="":
            continue
        try: 
            c=int(z)
            if c>len(rmsd)-1 or c<0:
                print("This index doesn't correspond to a cluster centre. Try again.\n")
                continue
            cluster_centers.append(c)
        except:
            print("Looks like you didn't give me an integer. Try again\n")

    #Assign cluster centers to cluster ids
    for i in range(0,len(cluster_centers)):
        point_i=cluster_centers[i]
        rhodelta.cluster[point_i]=i
    

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
        print(nnhd,nnhd_j,rhodelta.cluster[nnhd_j])
    
    #pickle rhodelta
    rhodelta.sort_values("point",inplace=True,ignore_index=True)
    rhodelta.to_pickle("rhodelta.pkl")
    return rhodelta


def dirty_clustering(traj,align_set, rmsd_set, rmsd_threshold):
    traj_align_idx=get_atomset_idx(traj,align_set)
    traj_rmsd_idx=get_atomset_idx(traj,rmsd_set)
    discarded=[]
    kept=[]
    for i in range(traj.n_frames):
        if (i in discarded):
            continue
        print("Processing frame %i" % i)
        frame_i=traj[i]
        crd_i=frame_i.xyz[0][traj_rmsd_idx]
        with pymp.Parallel() as p:
           for j in p.range(i,traj.n_frames):
               if (j in discarded):
                   continue
               frame_j=traj[j]
               frame_j=frame_j.superpose(reference=frame_i,atom_indices=traj_align_idx)
               crd_j=frame_j.xyz[0][traj_rmsd_idx]
               rmsd=np.sqrt(((((crd_i-crd_j)**2))*3).mean())
               #print(rmsd)
               if rmsd<rmsd_threshold:
                   discarded.append(j)
        kept.append(i)
    print(len(kept))
    print(len(discarded))
    return kept    

if __name__=="__main__":
   
   args=parse()
   print ("The following atoms will be used for alignment")
   align_set=get_atom_set(args.ref,args.align,args.input_gro)
   print ("The following atoms will be used for RMSD calculation")
   rmsd_set=get_atom_set(args.ref,args.rmsd,args.input_gro)
   traj=mdtraj.load(args.input_traj,top=args.input_gro,stride=args.stride)
   #rmsd=calc_rmsd(traj,args.ref, align_set, rmsd_set)
   #print(rmsd)
   #rmsd.to_pickle("rmsd.pkl")
   pairwise_rmsd=calc_pairwise_rmsd(traj,align_set,rmsd_set)
   rhodelta=laio(pairwise_rmsd,args.rmsd_threshold)
   

   #kept=dirty_clustering(traj,align_set,rmsd_set,args.rmsd_threshold)
   #traj[kept].save_xtc("kept.xtc")
   


   
   
   