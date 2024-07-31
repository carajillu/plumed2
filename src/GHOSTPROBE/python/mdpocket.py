import argparse
import mdtraj
import os
import sys
import subprocess
import numpy as np
import pymp
import pandas as pd

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('-f','--fpocket_input', nargs="?", help="holo input file for fpocket",default="holo.pdb")
    parser.add_argument('-hs','--holo_selection', nargs="?", help="selection to clean up holo structure",default="not water and not resname NA and not resname CL")
    parser.add_argument('-s','--ligand_sel', nargs="?", help="VMD-style selection for the ligand",default="resname LIG")
    parser.add_argument('-p','--topology', nargs="?", help="topology file to load trajectory with mdtraj",default="prod.gro")
    parser.add_argument('-t','--trajectory', nargs="?", help="Pre-aligned trajectory to analyse with mdpocket",default="prod.xtc")
    parser.add_argument('-r','--rmax', type=float, nargs="?", help="Distance from the ligand to considewr an alpha sphere part of the pocket of interest (nm)",default=0.3)
    args = parser.parse_args()
    return args

def get_pocket(holo_like,ligand,rmax):
    holo_like.save("holo_like.pdb")
    # Run fpocket
    subprocess.run(["fpocket","-f","holo_like.pdb"])
    # Load the pocket file
    load_name="holo_like_out/holo_like_out.pdb"
    fpocket_out=mdtraj.load(load_name)
    pocket_sel=fpocket_out.top.select("resname STP")
    pockets=fpocket_out.atom_slice(pocket_sel)
    pockets_xyz=pockets.xyz[0]
    #Generate an mdtraj object containing the atoms from pockets that are within rmax of ligand
    pocketname=f"pocket_{str(rmax)}.pdb"
    pocket_atoms=[]
    for i in range(len(pockets_xyz)):
        for j in range(len(ligand.xyz[0])):
            if np.linalg.norm(pockets_xyz[i]-ligand.xyz[0][j])<=rmax:
                pocket_atoms.append(i)
                break
    pocket_atoms=np.array(pocket_atoms)
    pocket=pockets.atom_slice(pocket_atoms)
    pocket.save(pocketname)
    return pocketname

def run_mdpocket(top_str,trj_str,pocket_str):
    top_obj=mdtraj.load(top_str)
    top_obj.save("top.pdb")
    mdpocket_cmd=["mdpocket",
                  "--trajectory_file",trj_str,
                  "--trajectory_format","xtc",
                  "-f","top.pdb",
                  "--selected_pocket",pocket_str]
    subprocess.run(mdpocket_cmd)
    
    sed_cmd = ["sed", "-i", "s/  */ /g", "mdpout_descriptors.txt"]
    subprocess.run(sed_cmd)

    return

if __name__=="__main__":
    
    args=parse()
    
    # Load fpocket input and separate ligand
    holo_sel=args.holo_selection
    holo=mdtraj.load(args.fpocket_input)
    holo=holo.atom_slice(holo.top.select(holo_sel))
    holo.save("holo.pdb")
    holo_like=holo.atom_slice(holo.top.select(f"not ({args.ligand_sel})"))
    ligand=holo.atom_slice(holo.top.select(args.ligand_sel))
    # Get the pocket
    pocket=get_pocket(holo_like,ligand,args.rmax)
    # Run mdpocket on the trajectory with pocket as the pocket of reference
    run_mdpocket(args.topology,args.trajectory,pocket)

