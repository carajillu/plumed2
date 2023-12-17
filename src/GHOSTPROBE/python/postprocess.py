import argparse
import mdtraj
import numpy as np
import pandas as pd
import sys

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_gro', nargs="?", help="GMX structure in gro or pdb format",default="traj.gro")
    parser.add_argument('-t','--input_traj', nargs="?", help="GMX trajectory in xtc or trr format",default="traj.xtc")
    parser.add_argument('-x','--input_xyz', nargs="?", help="protein file issued by plumed",default=None)
    parser.add_argument('-z','--stride', nargs="?", type=int,help="Load 1 in n frames",default=1)
    parser.add_argument('-n','--nprobes', nargs="?", type=int, help="Number of probes",default=1)
    parser.add_argument('-o','--output', nargs="?", help="Protein output in pdb format",default="protein_probes")
    parser.add_argument('-s','--subset', nargs="?", help="subset of atoms for when protein.xyz is not supplied (VMS style selection)",default="all")
    parser.add_argument('-a','--actmin', nargs="?", type=float, help="Minimum activity to print in the aligned pdb",default=1)
    parser.add_argument('-ts','--timestep', nargs="?", type=float, help="Time step for input trajectory (ps)",default=50)
    parser.add_argument('-b','--time_begin', nargs="?", type=float, help="Time to begin postprocessing (ps)",default=0)
    parser.add_argument('-e','--time_end', nargs="?", type=float, help="Time to end postprocessing",default=np.inf)

    args = parser.parse_args()
    return args

def process_xyz(input_xyz, snap_stride,snap_begin,snap_end):
    filein=open(input_xyz)
    line_id=0
    first_line=""
    second_line=""
    snap_id=0
    atom_crd=[]
    snap_crd=[]
    atomlist=[]
    
    for line in filein:
        if (line_id==0):
            first_line=line
            n_atoms=int(first_line)
        elif (line.startswith("Step") or (line.startswith("Probe"))): #too specific!
            pass
        elif (line_id>0 and line==first_line): #New snapshot: add last snapshot to atom_crd and reset snap_crd
            if (snap_id>=snap_begin and snap_id%snap_stride==0):
               atom_crd.append(snap_crd)
            snap_crd=[]
            snap_id=snap_id+1
            if (snap_id>=snap_end):
                break
        elif (snap_id<snap_begin):
            pass
        elif (snap_id%snap_stride==0):
            line=line.split()
            if (len(atomlist)<n_atoms):
                try: # if reading from protein.xyz
                   atom_id=int(line[0])-1
                except: # if reading from probe-0.xyz
                   atom_id=0
                atomlist.append(atom_id)
            crd_j=[float(line[1])/10,float(line[2])/10,float(line[3])/10]
            snap_crd.append(crd_j)
        else:
            pass
        line_id=line_id+1
        
    atom_crd=np.array(atom_crd)    
    return atomlist,atom_crd

def mktraj(xyz,id):
    top=mdtraj.Topology()
    chain=top.add_chain()
    resname="PRB" #so that we always have 3 digits
    residue=top.add_residue(resname,chain)
    top.add_atom("P"+str(id).zfill(2),mdtraj.element.helium,residue)
    trj=mdtraj.Trajectory(xyz,top)
    return trj

def stack_traj(protein_traj,probes_trj):
    for probe in probes_trj:
        protein_traj=protein_traj.stack(probe)
    return protein_traj

def get_activity(nprobes,stride,frame_begin,frame_end):
    activity=pd.DataFrame()
    for i in range(0,nprobes):
        name="P"+str(i).zfill(2)
        filename="probe-"+str(i)+"-stats.csv"
        activity[name]=pd.read_csv(filename,sep=" ").activity[frame_begin:frame_end:stride]
    activity.reset_index(drop=True, inplace=True)
    #print(activity)
    #sys.exit()    
    return activity

def print_probes_pdb(probes_trj,activity,activity_min):
    top=mdtraj.Topology()
    chain=top.add_chain()
    xyz=[]
    bfactors=[]
    for i in range(0,len(activity)):
        for j in range(0,len(activity.columns)):
            atomname=activity.keys()[j]
            if activity[atomname][i]<activity_min:
                continue
            residue=top.add_residue("PRB",chain)
            top.add_atom(atomname,mdtraj.element.helium,residue)
            xyz.append(probes_trj.xyz[i][j])
            bfactors.append(activity[atomname][i])
    trj=mdtraj.Trajectory(xyz,top)
    filename="probes_actmin_"+str(activity_min)+".pdb"
    trj.save_pdb(filename,bfactors=bfactors)
    return trj

if __name__=="__main__":

    args=parse()
    print(args)
    
    #process protein
    print("processing file "+args.input_gro)
    frame_begin=int(args.time_begin/args.timestep)
    if args.time_end!=np.inf:
        frame_end=int(args.time_end/args.timestep)
        print(frame_begin,frame_end,args.stride)
        traj_obj=mdtraj.load(args.input_traj,top=args.input_gro)[frame_begin:frame_end:args.stride]
        print(len(traj_obj))
    else:
        traj_obj=mdtraj.load(args.input_traj,top=args.input_gro)
        frame_end=len(traj_obj)
        print(frame_begin,frame_end,args.stride)
        traj_obj=traj_obj[frame_begin:frame_end:args.stride]
    
    if (args.input_xyz is not None):
       atomlist, xyz_prot=process_xyz(args.input_xyz,args.stride,frame_begin,frame_end)
       subset=traj_obj.atom_slice(atomlist)
       subset.xyz=xyz_prot
    else:
        atomlist=traj_obj.topology.select(args.subset)
        subset=traj_obj.atom_slice(atomlist)

    n_atoms=len(traj_obj.xyz[0])

    #process probes
    probes_trj=[]
    if args.nprobes>0:
        for i in range(args.nprobes):
            probefile="probe-"+str(i)+".xyz"
            print("reading file "+probefile)
            probeatomlist, xyz_probe=process_xyz(probefile,args.stride,frame_begin,frame_end)
            trj=mktraj(xyz_probe,i)
            probes_trj.append(trj)
    
    #join protein and probes in traj
    newtraj=stack_traj(subset,probes_trj)
    newtraj=newtraj.superpose(reference=newtraj[0],atom_indices=newtraj.topology.select("backbone"))
    #export
    try:
       newtraj[0].save_gro(args.output+".gro")
       newtraj.save_xtc(args.output+".xtc")
    except:
        print(args.output, "could not be saved")
    try:
        prot_traj=newtraj.atom_slice(newtraj.topology.select("not resname PRB"))
        prot_traj[0].save_gro("protein.gro")
        prot_traj.save_xtc("protein.xtc")
    except:
        print("protein traj could not be saved")
    
    probes_trj=newtraj.atom_slice(newtraj.topology.select("resname PRB"))
    activity=get_activity(args.nprobes,args.stride,frame_begin,frame_end)
    probes_trj=print_probes_pdb(probes_trj,activity,args.actmin)