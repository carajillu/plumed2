from os import wait
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
    parser.add_argument('-r','--ref', nargs="+", help="reference structure(s) gro or pdb",default=["ref.pdb"])
    parser.add_argument('-a','--align', nargs="?", help="atoms you want to align to reference (VMD style selection)",default="backbone")
    parser.add_argument('-d','--rmsd', nargs="?", help="atoms you want to align to reference (VMD style selection)",default="backbone")
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

def calc_rmsd(input_gro,input_traj,ref, align_set, rmsd_set):
    traj=mdtraj.load(input_traj,top=input_gro)
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
    rmsd=np.empty((traj.n_frames,traj.n_frames))
    with pymp.Parallel() as p:
       for i in p.range(traj.n_frames):
           print("processing frame %i" % i)
           frame_i=traj[i]
           crd_i=frame_i.xyz[0][traj_rmsd_idx]
           for j in range(i,traj.n_frames):
               if (i==j):
                   rmsd[i][j]=0
                   continue
               frame_j=traj[j]
               frame_j=frame_j.superpose(reference=frame_i,atom_indices=traj_align_idx)
               crd_j=frame_j.xyz[0][traj_rmsd_idx]
               rmsd[i][j]=np.sqrt(((((crd_i-crd_j)**2))*3).mean())
               rmsd[j][i]=rmsd[i][j]
    print("Max pairwise rmsd: ", np.max(rmsd))
    return rmsd

            


if __name__=="__main__":
   
   args=parse()
   print ("The following atoms will be used for alignment")
   align_set=get_atom_set(args.ref,args.align,args.input_gro)
   print ("The following atoms will be used for RMSD calculation")
   rmsd_set=get_atom_set(args.ref,args.rmsd,args.input_gro)
   rmsd=calc_rmsd(args.input_gro,args.input_traj,args.ref, align_set, rmsd_set)
   #print(rmsd)
   rmsd.to_pickle("rmsd.pkl")
   traj=mdtraj.load(args.input_traj,top=args.input_gro)
   pairwise_rmsd=calc_pairwise_rmsd(traj,align_set,rmsd_set)
   pickle.dump(pairwise_rmsd,open("pairwise.pkl","wb"))
   


   
   
   