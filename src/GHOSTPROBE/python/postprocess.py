import argparse
import mdtraj
import numpy as np
import pandas as pd
import sys

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_gro', nargs="?", help="GMX structure in gro or pdb format (default: traj.gro)",default="traj.gro")
    parser.add_argument('-t','--input_traj', nargs="?", help="GMX trajectory in xtc or trr format (default: traj.xtc)",default="traj.xtc")
    parser.add_argument('-r','--reference', nargs="?", help="Reference structure to align all frames to (default: ref.pdb)",default="ref.pdb")
    parser.add_argument('-rs','--ref_selection', nargs="?", help="VMD style selection for alignment to reference (default: backbone)",default="backbone")
    parser.add_argument('-x','--input_xyz', nargs="?", help="protein file issued by plumed (default: None)",default=None)
    parser.add_argument('-z','--stride', nargs="?", type=int,help="Load 1 in n frames (default: 1)",default=1)
    parser.add_argument('-n','--nprobes', nargs="?", type=int, help="Number of probes (default: 1)",default=1)
    parser.add_argument('-o','--output', nargs="?", help="Protein output in pdb format (default: protein_probes)",default="protein_probes")
    parser.add_argument('-s','--subset', nargs="?", help="subset of atoms for when protein.xyz is not supplied (VMD style selection) (default: all)",default="all")
    parser.add_argument('-a','--actmin', nargs="?", type=float, help="Minimum activity to print in the aligned pdb (default: 1)",default=1)
    parser.add_argument('-ts','--timestep', nargs="?", type=float, help="Time step for input trajectory (ps) (default: 50)",default=50)
    parser.add_argument('-b','--time_begin', nargs="?", type=float, help="Time to begin postprocessing (ps) (default: 0)",default=0)
    parser.add_argument('-e','--time_end', nargs="?", type=float, help="Time to end postprocessing (default: inf)",default=np.inf)

    args = parser.parse_args()
    return args

def process_xyz(input_xyz, stride, frame_begin, frame_end):
    print(f"Processing file: {input_xyz}")
    
    try:
        with open(input_xyz) as filein:
            line_id = 0
            atom_id = 0
            frame_id = 0
            atomlist = []
            crd_frame = []
            crd = []
            n_atoms = 0  # Initialize n_atoms to handle cases where the file might be empty or incorrect

            for line in filein:
                line = line.strip()  # Remove leading/trailing whitespace

                # Determine the number of atoms from the first line of each frame
                if line_id == 0 or (line_id % (n_atoms + 2) == 0):
                    try:
                        n_atoms = int(line)
                        frame_id += 1
                        #print(f"Starting frame {frame_id}")
                    except ValueError:
                        raise ValueError(f"Expected an integer for the number of atoms at line {line_id}, got '{line}'")
                
                # Skip the comment line (2nd line in each frame)
                elif line_id % (n_atoms + 2) == 1:
                    pass
                
                # Process atom data
                else:
                    line_split = line.split()
                    if len(line_split) < 4:
                        raise ValueError(f"Invalid atom line format at line {line_id}: '{line}'")

                    if len(atomlist) < n_atoms:
                        try:
                            atomlist.append(int(line_split[0]) - 1)
                        except ValueError: # Skip the atom ID if it's not an integer
                            pass

                    try:
                        crd_frame.append([
                            float(line_split[1]) / 10,
                            float(line_split[2]) / 10,
                            float(line_split[3]) / 10
                        ])
                    except ValueError:
                        raise ValueError(f"Invalid coordinate data at line {line_id}: '{line}'")

                    atom_id += 1

                    # End of frame, check if we need to store the frame
                    if line_id % (n_atoms + 2) == n_atoms + 1:
                        if frame_begin <= frame_id <= frame_end and (frame_id - frame_begin) % stride == 0:
                            crd.append(crd_frame)
                        if frame_id == frame_end:
                            break
                        crd_frame = []
                        atom_id = 0

                line_id += 1

            return atomlist, crd

    except FileNotFoundError:
        raise FileNotFoundError(f"File '{input_xyz}' not found.")

    except Exception as e:
        raise Exception(f"Unexpected error: {e}")


def mktraj(xyz,id):
    top=mdtraj.Topology()
    chain=top.add_chain()
    resname="PRB" #so that we always have 3 digits
    residue=top.add_residue(resname,chain)
    top.add_atom("P"+str(id).zfill(2),mdtraj.element.helium,residue)
    trj=mdtraj.Trajectory(xyz,top)
    print(trj)
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
        #print(frame_begin,frame_end,args.stride)
        traj_obj=mdtraj.load(args.input_traj,top=args.input_gro)[frame_begin:frame_end+1:args.stride]
        print(len(traj_obj))
    else:
        traj_obj=mdtraj.load(args.input_traj,top=args.input_gro)
        frame_end=len(traj_obj)
        #print(frame_begin,frame_end,args.stride)
        traj_obj=traj_obj[frame_begin:frame_end+1:args.stride]

    print(traj_obj)
    
    if (args.input_xyz is not None):
       atomlist, xyz_prot=process_xyz(args.input_xyz,args.stride,frame_begin,frame_end+1)
       subset=traj_obj.atom_slice(atomlist)
       subset.xyz=xyz_prot
    else:
        atomlist=traj_obj.topology.select(args.subset)
        subset=traj_obj.atom_slice(atomlist)
    print(subset)

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
    
    #join protein and probes in traj and align them to reference
    newtraj=stack_traj(subset,probes_trj)
    ref_obj=mdtraj.load(args.reference).atom_slice(atomlist)
    selection=ref_obj.topology.select(args.ref_selection)
    newtraj=newtraj.superpose(reference=ref_obj,atom_indices=selection,ref_atom_indices=selection)
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