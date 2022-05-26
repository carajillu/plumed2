import argparse
import mdtraj
import numpy as np

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_gro', nargs="?", help="GMX structure in gro or pdb format",default="traj.gro")
    parser.add_argument('-t','--input_traj', nargs="?", help="GMX trajectory in xtc or trr format",default="traj.xtc")
    parser.add_argument('-x','--input_xyz', nargs="?", help="protein file issued by plumed",default="protein.xyz")
    parser.add_argument('-p','--probe', nargs="?", help="probe file issued by plumed",default="probe-0.xyz")
    parser.add_argument('-o','--output', nargs="?", help="Protein output in pdb format",default="protein.pdb")

    args = parser.parse_args()
    return args

def process_protein(input_xyz):
    atomlist=[]
    atom_crd=[]
    snap_crd=[]
    filein=open(input_xyz)
    for line in filein:
        if line.startswith("Step"):
            continue
        line=line.split()
        if len(line)==1:
            n_atoms=int(line[0])
            continue
        if (len(atomlist)<n_atoms):
            atomlist.append(int(line[0])-1) # we are getting just list indices
        crd_j=[float(line[1])/10,float(line[2])/10,float(line[3])/10]
        snap_crd.append(crd_j)
        if len(snap_crd)==n_atoms:
            atom_crd.append(snap_crd)
            snap_crd=[]
    filein.close()

    atom_crd=np.array(atom_crd)
    return atomlist, atom_crd

if __name__=="__main__":

    args=parse()
    traj_obj=mdtraj.load(args.input_traj,top=args.input_gro)
    atomlist, xyz_arr=process_protein(args.input_xyz)
    
    subset=traj_obj.atom_slice(atomlist)
    subset.xyz=xyz_arr
    subset.save_pdb(args.output)
