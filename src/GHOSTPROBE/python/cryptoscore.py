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
    parser.add_argument('--cleanup', action=argparse.BooleanOptionalAction,help="Remove fpocket output files after processing")
    parser.add_argument('-f','--topology', nargs="?", help="Topology file for mdtraj",default="protein.pdb")
    parser.add_argument('-e','--trj_eq_path', nargs=1, help="Equilibrium MD trajectory file for mdtraj",default="equilibrium.xtc")
    parser.add_argument('-es','--equilibrium_scores', nargs=1, help="Equilibrium MD trajectory file for mdtraj",default="equilibrium_scores.pdb")
    parser.add_argument('-b','--trj_bias_path', nargs="+", help="Biased MD trajectory file(s) for mdtraj",default=["biased.xtc"])
    parser.add_argument('-o','--output', nargs="?", help="Output PDB file with the cryproscore of each atom as B-factor",default="crypto.pdb")
    parser.add_argument('--r_min', type=float, help="Minimum distance for S_off function",default=0.3076)
    parser.add_argument('--delta_r', type=float, help="Distance over which S_off turns off",default=0.0564)
    parser.add_argument('--drug_min', type=float, help="Minimim druggability score to include a pocket in the analysis",default=0.1)
    args = parser.parse_args()
    return args

def S_off(r,r_min,delta_r):
    m=(r-r_min)/delta_r
    if m<0:
        return 1
    elif m>1:
        return 0
    else:
        return 3*(m-1)**4+2*(m-1)**6
    
def get_druggable_pockets(info,dmin):
    druggable_pockets=[]
    with open(info) as filein:
        pocket_id=0 # there's no pocket 0 in fpocket output
        drugscore=0
        for line in filein:
            if line.startswith("Pocket"):
                pocket_id=int(line.split()[1])
            elif "Druggability Score" in line:
                drugscore=float(line.split()[3])
                if drugscore>=dmin:
                    druggable_pockets.append(pocket_id)
    return druggable_pockets

def run_fpocket(trj,dmin):
    # Run fpocket on the trajectory
    # out_list = pymp.shared.array((len(trj),), dtype='str') # this returns an error
    out_list=[]

    with pymp.Parallel() as p:
        for i in p.range(len(trj)):
            frame_name=f"frame_{i}"
            frame_pdb=f"{frame_name}.pdb"
            out_dir=f"{frame_name}_out"
            out_info=f"{out_dir}/{frame_name}_info.txt"
            out_file=f"{out_dir}/{frame_name}_drug.pdb"
            if not os.path.isfile(out_file):
                trj[i].save_pdb(frame_pdb)
                try:
                    print(f"Running fpocket on {frame_pdb} on thread {p.thread_num}")
                    cmd = f"fpocket -f {frame_pdb}"
                    subprocess.run(cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Error running fpocket: {e}")
                druggable_pockets=get_druggable_pockets(out_info,dmin)
                cmd=f"grep ATOM {frame_pdb} >> {out_file}"
                subprocess.run(cmd, shell=True, check=True)
                for pocket_id in druggable_pockets:
                    pocket_name=f"{out_dir}/pockets/pocket{str(pocket_id)}_vert.pqr"
                    cmd=f"grep ATOM {pocket_name} >> {out_file}"
                    subprocess.run(cmd, shell=True, check=True)

    # This could be part of the earlier loop, but I can't create a shared array of type str
    for i in range(len(trj)):
          frame_name=f"frame_{i}"
          frame_pdb=f"{frame_name}.pdb"
          out_dir=f"{frame_name}_out"
          out_info=f"{out_dir}/{frame_name}_info.txt"
          out_file=f"{out_dir}/{frame_name}_drug.pdb"  
          if os.path.isfile(out_file):
            out_list.append(out_file)
          else:
            print(f"Could not find {out_file}. Check filename in code.")
            sys.exit()
    
    return out_list

def get_scores(pdb,r_min,delta_r):
    struct_obj=mdtraj.load(pdb)
    stp_idx=struct_obj.top.select("resname STP")
    protein_idx=struct_obj.top.select("protein")
    scores = pymp.shared.array((len(protein_idx),), dtype='float64')
    crd=struct_obj.xyz[0]
    
    with pymp.Parallel() as p:
        for i in p.range(len(protein_idx)):
            score_i=0
            for j in range(len(stp_idx)):
                r=np.linalg.norm(crd[protein_idx[i]]-crd[stp_idx[j]])
                score_i+=S_off(r,r_min,delta_r)
            scores[i]=score_i
    return scores


def calc_pocketscores(pdb_list,ref_structure,r_min,delta_r):
    n_atoms=ref_structure.n_atoms
    n_frames=len(pdb_list)
    pocketscores=[0]*n_atoms
    for pdb in pdb_list:
        pocketscores_pdb=get_scores(pdb,r_min,delta_r)
        for i in range(0,n_atoms):
            pocketscores[i]+=pocketscores_pdb[i]/(n_frames*100) # to keep within b-factor range
    return pocketscores
                    
def calc_cryptoscores(pocketscores_eq,pocketscores_bias):
    cryptoscores=[]
    for i in range(0,len(pocketscores_eq)):
        cryptoscores.append(pocketscores_bias[i]-pocketscores_eq[i])
    return cryptoscores

def output_score_pdb(trj_obj,scores,out_pdb):
    trj_obj.save_pdb(out_pdb,bfactors=scores)
    return

def mdtraj_get_atoms(trj_obj):
    atomnames=[]
    for atom in trj_obj.topology.atoms:
        atomnames.append(atom)
    return atomnames

def mdtraj_get_residues(trj_obj):
    residnames=[]
    for atom in trj_obj.topology.atoms:
        residnames.append(atom.residue)
    return residnames

def pocketscores_byres(z):
    resnames=z["residue"].unique()
    res_scores=pd.DataFrame()
    res_scores["residue"]=resnames
    res_scores["equilibrium"]=0
    for bias in z.columns[2:]:
        res_scores[bias]=0
    for res in resnames:
        res_scores.loc[res_scores["residue"]==res,"equilibrium"]=sum(z.loc[z["residue"]==res,"equilibrium"])
        for bias in z.columns[2:]:
            res_scores.loc[res_scores["residue"]==res,bias]=sum(z.loc[z["residue"]==res,bias])
    return res_scores


if __name__=="__main__":
    root_dir=os.getcwd()
    args=parse()
    ref_obj=mdtraj.load(args.topology)

    z=pd.DataFrame()
    z["atom"]=mdtraj_get_atoms(ref_obj)
    z["residue"]=mdtraj_get_residues(ref_obj)

    if os.path.isfile(args.equilibrium_scores):
        pocketscores_eq=get_scores(args.equilibrium_scores)
    else:
        if args.debug:
           eq_trj=mdtraj.load(args.trj_eq_path,top=args.topology)[0:1]
        else:
           eq_trj=mdtraj.load(args.trj_eq_path,top=args.topology)
        os.makedirs("equilibrium",exist_ok=True)
        os.chdir("equilibrium")
        os.makedirs("fpocket",exist_ok=True)
        os.chdir("fpocket")
        fpocketlist_eq=run_fpocket(eq_trj,args.drug_min)
        pocketscores_eq=calc_pocketscores(fpocketlist_eq,ref_obj,args.r_min,args.delta_r)
        z["equilibrium"]=pocketscores_eq
        os.chdir("..")
        outname=args.equilibrium_scores
        output_score_pdb(ref_obj[0],pocketscores_eq,args.equilibrium_scores)
        os.chdir(root_dir)
    
    for bias_path in args.trj_bias_path:
        print(bias_path)
        if args.debug:
           bias_obj=mdtraj.load(bias_path,top=args.topology)[0:1]
        else:
           bias_obj=mdtraj.load(bias_path,top=args.topology)
        dirname=bias_path.split(".")[0]
        os.makedirs(dirname,exist_ok=True)
        os.chdir(dirname)
        os.makedirs("fpocket",exist_ok=True)
        os.chdir("fpocket")
        fpocketlist=run_fpocket(bias_obj,args.drug_min)
        pocketscores_bias=calc_pocketscores(fpocketlist,bias_obj,args.r_min,args.delta_r)
        z[bias_path]=pocketscores_bias
        os.chdir("..")
        outname=dirname+"_pocketscores.pdb"
        output_score_pdb(bias_obj[0],pocketscores_bias,outname)
        cryptoscores=calc_cryptoscores(pocketscores_eq,pocketscores_bias)
        outname=dirname+"_crypto.pdb"
        output_score_pdb(bias_obj[0],cryptoscores,outname)
        os.chdir(root_dir)
    
    z.to_csv("atom_scores.csv",index=False)
    z=pocketscores_byres(z)
    z.to_csv("resid_scores.csv",index=False)
    
    
    
    