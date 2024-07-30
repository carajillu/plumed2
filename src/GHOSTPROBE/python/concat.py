import os
import sys
import pandas as pd
import mdtraj
import argparse
import glob
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noplumed', action=argparse.BooleanOptionalAction)
    parser.add_argument('-p','--top', nargs="?", help="Topology file for mdtraj",default="prod_0/prod.gro")
    parser.add_argument('-t','--xtc', nargs="?", help="Name of the RAW Gromacs trajectory files (must be the same in all directories)",default="prod.xtc")
    parser.add_argument('-a','--pattern', nargs="?", help="pattern matching the directory names. Simulation index (starting at 0) will be appended at the end.",default="prod_")
    parser.add_argument('-r','--reference', nargs="?", help="Reference structure for alignement. If not supplied, will align to first frame.",default=None)
    parser.add_argument('-rs','--ref_selection', nargs="?", help="subset of atoms to align to reference (VMD style selection)",default="backbone")
    parser.add_argument('-os','--out_selection', nargs="?", help="",default="not water")
    parser.add_argument('-b', '--nsim', nargs="?", type=int, help="Number of simulations sto concatenate (MUST start at 0, be consecutive and follow a pattern)",default=1)
    parser.add_argument('-x','--input_xyz', nargs="?", help="protein file issued by plumed (must be the same in all directories)",default="protein.xyz")
    parser.add_argument('-n','--nprobes', nargs="?", type=int, help="Number of probes (must be the same in all directories)",default=1)
    args = parser.parse_args()
    return args

def concat_xtc(topology,names,ref_obj,align_sel,output_sel,outfile):
    traj=mdtraj.load(names[0],top=topology)
    selection=get_selection(ref_obj,traj,align_sel)
    print(names[0],traj)
    for i in range(1,len(names)):
        try:
          traj_i=mdtraj.load(names[i],top=topology)
          time_offset = traj.time[-1]
          traj_i.time += time_offset
          print(names[i],traj_i)
          traj=mdtraj.join([traj,traj_i])
        except:
          print(f"Could not load {names[i]}. Skipping.")
          continue
    traj.image_molecules(inplace=True)
    traj.superpose(reference=ref_obj[0],atom_indices=selection,ref_atom_indices=selection)
    traj=traj.atom_slice(traj.topology.select(output_sel))
    traj.save_xtc(outfile)
    traj[0].save_gro(outfile.replace(".xtc",".gro"))
    print(traj)
    return


def concat_xyz(names,outfile):
    cmd=f"cat {names[0]} > {outfile}"
    os.system(cmd)
    for i in range(1,len(names)):
        cmd=f"cat {names[i]} >> {outfile}"
        try:
           os.system(cmd)
        except:
           print(f"Could not concatenate {names[i]}. Skipping.")
           continue
    return

def concat_df(names,outfile):
    df=pd.read_csv(names[0],sep=" ")
    for i in range(1,len(names)):
        try:
           df_i=pd.read_csv(names[i],sep=" ")
           df=pd.concat([df,df_i],ignore_index=True)
        except:
            print(f"Could not concatenate {names[i]}. Skipping.")
            continue
    df.to_csv(outfile,sep=" ", index=False)
    return df

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

          

if __name__=="__main__":
    args=parse()

    os.makedirs("concat",exist_ok=True)

    
    xtc_lst=[]
    for i in range(0,args.nsim):
        xtc_lst.append(args.pattern+str(i)+"/"+args.xtc)

    print(args.top)
    print(xtc_lst)
    if args.reference is not None:
        ref_obj=mdtraj.load(args.reference)[0]
    else:
        ref_obj=mdtraj.load(xtc_lst[0],top=args.top)[0]
    xtc=concat_xtc(topology=args.top,names=xtc_lst,ref_obj=ref_obj,align_sel=args.ref_selection,output_sel=args.out_selection,outfile="concat/"+args.xtc)
    
    if args.noplumed:
        sys.exit("The --noplumed flag has been passed. Will not process probes and protein files. Exiting.")
    
    protein_lst=[]
    for i in range(0,args.nsim):
        protein_lst.append(args.pattern+str(i)+"/"+args.input_xyz)
    print(protein_lst)
    protein=concat_xyz(names=protein_lst,outfile="concat/"+args.input_xyz)


    for i in range(0,args.nprobes):
        probe_i_lst=[]
        stats_i_lst=[]
        for j in range(0,args.nsim):
            probe_i_lst.append(args.pattern+str(j)+"/probe-"+str(i)+".xyz")
            stats_i_lst.append(args.pattern+str(j)+"/probe-"+str(i)+"-stats.csv")
        probe_i=concat_xyz(names=probe_i_lst,outfile="concat/probe-"+str(i)+".xyz")
        stats_i=concat_df(names=stats_i_lst,outfile="concat/probe-"+str(i)+"-stats.csv")
        print(probe_i_lst)


    

