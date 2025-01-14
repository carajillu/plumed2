import os
import sys
import pandas as pd
import mdtraj
import argparse
import glob
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--top', nargs="?", help="Topology file for mdtraj",default="prod_0/prod.gro")
    parser.add_argument('-t','--xtc', nargs="?", help="Name of the RAW Gromacs trajectory files (must be the same in all directories)",default="prod.xtc")
    parser.add_argument('-s','--ndir', nargs="?", help="Number of simulation directories",default=1)
    parser.add_argument('-a','--pattern', nargs="?", help="pattern matching the directory names. Simulation index (starting at 0) will be appended at the end.",default="prod_")
    parser.add_argument('-b', '--nsim', nargs="?", type=int, help="Number of simulations sto concatenate (MUST start at 0, be consecutive and follow a pattern)",default=1)
    parser.add_argument('-x','--input_xyz', nargs="?", help="protein file issued by plumed (must be the same in all directories)",default="protein.xyz")
    parser.add_argument('-n','--nprobes', nargs="?", type=int, help="Number of probes (must be the same in all directories)",default=1)
    args = parser.parse_args()
    return args

def concat_xtc(topology,names,outfile):
    traj=mdtraj.load(names[0],top=topology)
    print(names[0],traj)
    for i in range(1,len(names)):
        traj_i=mdtraj.load(names[i],top=topology)
        time_offset = traj.time[-1]
        traj_i.time += time_offset
        print(names[i],traj_i)
        traj=mdtraj.join([traj,traj_i])
    traj.save_xtc(outfile)
    print(traj)
    return


def concat_xyz(names,outfile):
    cmd=f"cat {names[0]} > {outfile}"
    os.system(cmd)
    for i in range(1,len(names)):
        cmd=f"cat {names[i]} >> {outfile}"
        os.system(cmd)
    return

def concat_df(names,outfile):
    df=pd.read_csv(names[0],sep=" ")
    for i in range(1,len(names)):
        df_i=pd.read_csv(names[i],sep=" ")
        df=pd.concat([df,df_i],ignore_index=True)
    df.to_csv(outfile,sep=" ", index=False)
    return df
          

if __name__=="__main__":
    args=parse()

    os.makedirs("concat",exist_ok=True)

    
    xtc_lst=[]
    for i in range(0,args.nsim):
        xtc_lst.append(args.pattern+str(i)+"/"+args.xtc)

    print(args.top)
    print(xtc_lst)
    xtc=concat_xtc(topology=args.top,names=xtc_lst,outfile="concat/"+args.xtc)
    
    
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


    

