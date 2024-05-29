import os
import argparse
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir', nargs="?", help="Directory to get the restart files from",default="../run_0")
    parser.add_argument('-x','--input_xyz', nargs="?", help="protein file issued by plumed",default="protein.xyz")
    parser.add_argument('-n','--nprobes', nargs="?", type=int, help="Number of probes",default=1)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args=parse()
    with open(f"{args.dir}/{args.input_xyz}","r") as f:
        for line in f:
            line=line.split()
            n=int(line[0])+2
            break
    cmd=f"tail -n {n} {args.dir}/{args.input_xyz} > protein.xyz"
    os.system(cmd)
    for i in range(0,args.nprobes):
        id=str(i)
        cmd=f"tail -n 3 {args.dir}/probe-{id}.xyz > probe-{id}.xyz"
        os.system(cmd)
    

