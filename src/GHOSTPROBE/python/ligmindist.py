import math
import mdtraj
import pandas as pd
import numpy as np
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gl','--gro_lig', nargs="?", help="*.gro file of the liganded protein (default=prod_dry.gro)",default="prod_dry.gro")
    parser.add_argument('-xl','--xtc_lig', nargs="?", help="*.xtc file of the liganded trajectory (default=prod_dry.xtc)",default="prod_dry.xtc")
    parser.add_argument('-nl','--name_lig', nargs="?", help="residue name of the ligand (default=BNZ)",default="BNZ")
    parser.add_argument('-gp','--gro_prb', nargs="?", help="*.gro file of the protein with ghost probes (default=protein_probes.gro)",default="protein_probes.gro")
    parser.add_argument('-xp','--xtc_prb', nargs="?", help="*.xtc file of the trajectory with ghost probes (default=protein_probes.xtc)",default="protein_probes.xtc")
    parser.add_argument('-np','--name_prb', nargs="?", help="residue name of the ghost probes (default=PRB)",default="PRB")
    parser.add_argument('-p','--nprobes', nargs="?", help="Number of ghost probes (default=16)",default=16)
    parser.add_argument('-a','--min_activity', nargs="?", help="Minimum psi_{i} to take into account (default=1)",default=1.0)

    args = parser.parse_args()
    return args

def get_activity(nprobes):
    activity=pd.DataFrame()
    for i in range(0,nprobes):
        csv="probe-"+str(i)+"-stats.csv"
        z=pd.read_csv(csv,sep=" ")
        key="P"+str(i).zfill(2)
        activity[key]=z.activity
    return activity

def get_new_trj(trj,sel_str):
    sel=trj.topology.select(sel_str)
    new_trj=trj.atom_slice(sel)
    return new_trj
            
def calc_mind(activity,min_activity,trj_lig,trj_prb):
    mind_lst=[]
    nframes=trj_lig.n_frames
    for k in range(0,nframes):
        for i in range(0,len(activity.keys())):
            key=activity.keys()[i]
            a=activity[key][k]
            if (a<min_activity):
               continue
            xyz_prb=trj_prb.xyz[k][i]
            xyz_lig=trj_lig.xyz[k]
            mind=np.inf
            for atom_xyz in xyz_lig:
                #r=math.sqrt((atom_xyz[0]-xyz_prb[0])**2+(atom_xyz[1]-xyz_prb[1])**2+(atom_xyz[2]-xyz_prb[2])**2)
                r=np.linalg.norm(atom_xyz-xyz_prb)
                if (r<mind):
                   mind=r
            mind_lst.append(mind)
    mind_df=pd.DataFrame()
    mind_df["mind"]=mind_lst
    mind_df.to_csv("mind.csv",index=False)
    return mind_df


if __name__=="__main__":
    args=parse()
    name_lig=args.name_lig
    trj_lig=mdtraj.load(args.xtc_lig,top=args.gro_lig)
    trj_lig=trj_lig.superpose(reference=trj_lig[0],atom_indices=trj_lig.topology.select("backbone"))
    ligsel_str="resname "+args.name_lig+" and mass 10 to 100"
    trj_lig=get_new_trj(trj_lig,ligsel_str)
 
    trj_prb=mdtraj.load(args.xtc_prb,top=args.gro_prb)
    trj_prb=trj_prb.superpose(reference=trj_prb[0],atom_indices=trj_prb.topology.select("backbone"))
    prbsel_str="resname "+args.name_prb
    trj_prb=get_new_trj(trj_prb,"resname PRB")

    activity=get_activity(args.nprobes)
    
    mind_lst=calc_mind(activity,args.min_activity,trj_lig,trj_prb)
