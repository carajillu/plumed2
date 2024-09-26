import argparse
import subprocess
import mdtraj
import numpy as np
import os
import pymp
import pandas as pd

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--ref_structure', nargs="?", help="Reference holo structure",default="holo.pdb")
    parser.add_argument('-rs','--ref_selection', nargs="?", help="Selection for alignement of the trajectory to the reference structure",default="backbone")
    parser.add_argument('-ls','--lig_selection', nargs="?", help="Selection for the ligand in the reference structure",default="resname LIG")
    parser.add_argument('-f','--topology', nargs="?", help="Topology for mdtraj (pdb/gro)",default="prod.gro")
    parser.add_argument('-x','--trajectory', nargs="?", help="Trajectory for mdtraj (xtc/trr)",default="prod.xtc")
    parser.add_argument('-c','--cutoff', nargs="?",type=float, help="Pocket-ligand cutoff",default=0.3)
    args = parser.parse_args()
    return args

def get_ligand_obj(ref_obj,lig_selection,hydrogens=False):
    ligand_sel_raw=ref_obj.topology.select(lig_selection)
    if not hydrogens:
        ligand_sel=[i for i in ligand_sel_raw if ref_obj.topology.atom(i).element.symbol!="H"]
    else:
        ligand_sel=ligand_sel_raw
    ligand_obj=ref_obj.atom_slice(ligand_sel)
    return ligand_obj

def get_selection(ref_obj,trj_obj,sel_str):
    selection=ref_obj.topology.select(sel_str)
    selection_trj=trj_obj.topology.select(sel_str)
    assert len(selection)==len(selection_trj), "Selections have different number of atoms"
    for i in range(0,len(selection)):
        ref_name=ref_obj.topology.atom(selection[i]).name
        trj_name=trj_obj.topology.atom(selection_trj[i]).name
        assert ref_name==trj_name, f"Selections have different atoms at position {selection[i]}: {ref_name} vs {trj_name}"
    print("assert selection OK")
    return selection

def align_trj(trj_obj,ref_obj,selection):
    anchor_molecules=[set(trj_obj.topology.residue(0).atoms)]
    trj_obj.image_molecules(inplace=True,anchor_molecules=anchor_molecules)
    trj_obj.superpose(reference=ref_obj,atom_indices=selection,ref_atom_indices=selection)
    return trj_obj

def select_pocket(mdpocket_out,ligand_obj,cutoff):
    pocket_obj=mdtraj.load(mdpocket_out)
    #select the pocket atoms that are within a cutoff distance from at least one ligand atom
    pocket_sel=[]
    for i in range(0,len(pocket_obj.xyz[0])):
        xyz_i=pocket_obj.xyz[0][i]
        for j in range(0,len(ligand_obj.xyz[0])):
            xyz_j=ligand_obj.xyz[0][j]
            dist=np.linalg.norm(xyz_i-xyz_j)
            if dist<cutoff:
                pocket_sel.append(i)
                break
    new_pocket_obj=pocket_obj.atom_slice(pocket_sel)
    new_pocket_obj.save_pdb("pocket.pdb")
    n_atoms=new_pocket_obj.n_atoms
    return n_atoms

if __name__ == "__main__":

    args = parse()

    # Get ligand object from reference structure
    ref_obj=mdtraj.load(args.ref_structure)
    ligand_obj=get_ligand_obj(ref_obj,args.lig_selection)

    # Load trajectory and reference structure, align trajectory to reference structure
    trj_obj=mdtraj.load(args.trajectory,top=args.topology)
    selection=get_selection(ref_obj,trj_obj,args.ref_selection)
    print(selection)
    trj_obj=align_trj(trj_obj,ref_obj,selection)

    
    pock_vol=[]

    with open("pdb_list_file","w") as f:
        f.write("tmp.pdb\n")

    volume=pymp.shared.array((len(trj_obj),), dtype='float64')
    alphaspheres=pymp.shared.array((len(trj_obj),), dtype='int32')
    root_dir=os.getcwd()
    with pymp.Parallel() as p:
        for i in p.range(0, len(trj_obj)):
            print(f"processing frame {i}")
            os.makedirs(f"frame_{i}",exist_ok=True)
            os.chdir(f"frame_{i}")
            trj_obj[i].save_pdb("tmp.pdb")
            subprocess.run(["mdpocket", "--pdb_list", "../pdb_list_file"])
            n_alpha=select_pocket("mdpout_freq_iso_0_5.pdb", ligand_obj, args.cutoff)
            alphaspheres[i]=n_alpha
            if n_alpha == 0:
                volume[i]=0
                os.chdir(root_dir)
                continue
            subprocess.run(["mdpocket", "--pdb_list", "../pdb_list_file", "--selected_pocket", "pocket.pdb"])
            with open("mdpout_descriptors.txt") as f:
                for line in f:
                    if line.startswith("snapshot"):
                        continue
                    else:
                        volume[i]=float(line.split()[1])
            os.chdir(root_dir)
            if n_alpha > 0:
               subprocess.run(["rm", "-r", f"frame_{i}"])
    subprocess.run(["rm", "pdb_list_file"])

    snapshot_id=np.arange(0,len(trj_obj))
    volumes=pd.DataFrame({"snapshot":snapshot_id,"volume":volume,"n_alpha":alphaspheres})
    volumes.to_csv("volumes.csv",sep=" ",index=False)