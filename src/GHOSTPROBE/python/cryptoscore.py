import argparse
import mdtraj
import os
import sys
import subprocess

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--topology', nargs="?", help="Topology file for mdtraj",default="protein.pdb")
    parser.add_argument('-e','--trj_eq_path', nargs=1, help="Equilibrium MD trajectory file for mdtraj",default="equilibrium.xtc")
    parser.add_argument('-es','--equilibrium_scores', nargs=1, help="Equilibrium MD trajectory file for mdtraj",default="equilibrium_scores.pdb")
    parser.add_argument('-b','--trj_bias_path', nargs="+", help="Biased MD trajectory file(s) for mdtraj",default=["biased.xtc"])
    parser.add_argument('-o','--output', nargs="?", help="Output PDB file with the cryproscore of each atom as B-factor",default="crypto.pdb")
    args = parser.parse_args()
    return args

def run_fpocket(trj):
    # Run fpocket on the trajectory
    out_list=[]
    for i in range(0,len(trj)):
        frame_pdb=f"frame_{i}.pdb"
        trj[i].save_pdb(frame_pdb)
        try:
            cmd = f"fpocket -f {frame_pdb}"
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error running fpocket: {e}")
        out_file=f"frame_{i}_out/frame_{i}_out.pdb"
        if os.path.isfile(out_file):
           out_list.append(out_file)
        else:
            print(f"Could not find {out_file}. Check filename in code.")
            sys.exit()
    return out_list

def get_scores(pdb):
    scores=[]
    with open(pdb) as f:
        for line in f:
            if "ATOM" in line:
                line=line.split()
                scores.append(float(line[10])) # B-factor is the 11th column
            if "HETATM" in line:
                break
    return scores


def calc_pocketscores(pdb_list,ref_structure):
    n_atoms=ref_structure.n_atoms
    n_frames=len(pdb_list)
    pocketscores=[0]*n_atoms
    for pdb in pdb_list:
        pocketscores_pdb=get_scores(pdb)
        for i in range(0,n_atoms):
            pocketscores[i]+=pocketscores_pdb[i]/n_frames
    return pocketscores
                    
def calc_cryptoscores(pocketscores_eq,pocketscores_bias):
    cryptoscores=[]
    for i in range(0,len(pocketscores_eq)):
        cryptoscores.append(pocketscores_bias[i]-pocketscores_eq[i])
    return cryptoscores

def output_score_pdb(ref_obj,cryptoscores,out_pdb):
    for residue in ref_obj.top.residues:
        resid_score=0
        ca_idx=None
        for atom in residue.atoms:
            if atom.name=="CA":
                ca_idx=atom.index
            resid_score+=cryptoscores[atom.index]/residue.n_atoms
        if ca_idx is not None:
           cryptoscores[ca_idx]=resid_score
    ref_obj.save_pdb(out_pdb,bfactors=cryptoscores)
    return

if __name__=="__main__":
    root_dir=os.getcwd()
    args=parse()
    ref_obj=mdtraj.load(args.topology)
    if os.path.isfile(args.equilibrium_scores):
        pocketscores_eq=get_scores(args.equilibrium_scores)
    else:
        eq_trj=mdtraj.load(args.trj_eq_path,top=args.topology)
        os.mkdir("equilibrium")
        os.chdir("equilibrium")
        fpocketlist_eq=run_fpocket(eq_trj)
        pocketscores_eq=calc_pocketscores(fpocketlist_eq,ref_obj)
        outname=args.equilibrium_scores
        output_score_pdb(ref_obj,pocketscores_eq,args.equilibrium_scores)
        os.chdir(root_dir)
    
    for bias_path in args.trj_bias_path:
        print(bias_path)
        bias_obj=mdtraj.load(bias_path,top=args.topology)
        dirname=bias_path.split(".")[0]
        os.mkdir(dirname)
        os.chdir(dirname)
        fpocketlist=run_fpocket(bias_obj)
        pocketscores_bias=calc_pocketscores(fpocketlist,ref_obj)
        outname=dirname+"_pocketscores.pdb"
        output_score_pdb(ref_obj,pocketscores_bias,outname)
        cryptoscores=calc_cryptoscores(pocketscores_eq,pocketscores_bias)
        outname=dirname+"_crypto.pdb"
        output_score_pdb(ref_obj,cryptoscores,outname)
        os.chdir(root_dir)
    
    
    
    