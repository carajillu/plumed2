import argparse
import mdtraj
import numpy as np

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_gro', nargs="?", help="GMX structure in gro or pdb format",default="traj.gro")
    parser.add_argument('-t','--input_traj', nargs="?", help="GMX trajectory in xtc or trr format",default="traj.xtc")
    parser.add_argument('-x','--input_xyz', nargs="?", help="protein file issued by plumed",default=None)
    parser.add_argument('-n','--nprobes', nargs="?", type=int, help="Number of probes",default=1)
    parser.add_argument('-o','--output', nargs="?", help="Protein output in pdb format",default="protein")
    parser.add_argument('-s','--subset', nargs="?", help="subset of atoms for when protein.xyz is not supplied (VMS style selection)",default="all")

    args = parser.parse_args()
    return args

def process_xyz(input_xyz):
    atomlist=[]
    atom_crd=[]
    snap_crd=[]
    filein=open(input_xyz)
    for line in filein:
        if line.startswith("Step") or line.startswith("Probe"):
            continue
        line=line.split()
        if len(line)==1:
            n_atoms=int(line[0])
            continue
        if (len(atomlist)<n_atoms):
            try:
               atomlist.append(int(line[0])-1) # we are getting just list indices
            except:
               pass # for when we process probes
        crd_j=[float(line[1])/10,float(line[2])/10,float(line[3])/10]
        snap_crd.append(crd_j)
        if len(snap_crd)==n_atoms:
            atom_crd.append(snap_crd)
            snap_crd=[]
    filein.close()

    atom_crd=np.array(atom_crd)
    return atomlist, atom_crd

def mktraj(xyz,id):
    top=mdtraj.Topology()
    chain=top.add_chain()
    resname="PRB" #so that we always have 3 digits
    residue=top.add_residue(resname,chain)
    top.add_atom("P"+str(id).zfill(2),mdtraj.element.germanium,residue)
    trj=mdtraj.Trajectory(xyz,top)
    return trj

def stack_traj(protein_traj,probes_trj):
    for probe in probes_trj:
        protein_traj=protein_traj.stack(probe)
    return protein_traj

if __name__=="__main__":

    args=parse()
    
    #process protein
    print("processing file "+args.input_gro)
    traj_obj=mdtraj.load(args.input_traj,top=args.input_gro)
    if (args.input_xyz is not None):
       atomlist, xyz_prot=process_xyz(args.input_xyz)
       subset=traj_obj.atom_slice(atomlist)
       subset.xyz=xyz_prot
    else:
        atomlist=traj_obj.topology.select(args.subset)
        subset=subset=traj_obj.atom_slice(atomlist)

    #process probes
    probes_trj=[]
    if args.nprobes>0:
        for i in range(args.nprobes):
            probefile="probe-"+str(i)+".xyz"
            print("reading file "+probefile)
            probeatomlist, xyz_probe=process_xyz(probefile)
            trj=mktraj(xyz_probe,i)
            probes_trj.append(trj)
    
    #join protein and probes in traj
    newtraj=stack_traj(subset,probes_trj)
    newtraj=newtraj.superpose(reference=newtraj[0],atom_indices=newtraj.topology.select("backbone"))
    #export
    newtraj[0].save_gro(args.output+".gro")
    newtraj.save_xtc(args.output+".xtc")

