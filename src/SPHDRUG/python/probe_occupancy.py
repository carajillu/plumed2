import pandas as pd
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_pdb', nargs="?", help="input GMX structure in gro or pdb format",default="traj.pdb")
    parser.add_argument('-o','--output_pdb', nargs="?", help="input GMX structure in pdb format",default="traj.pdb")
    parser.add_argument('-c','--input_csv', nargs="+", help="stats file(s) from plumed",default=["probe-0-stats.csv"])
    parser.add_argument('-t','--activity_threshold',nargs="?", type=float, help="minimum activity for a probe to be printed",default=0.75)
    args = parser.parse_args()
    return args

def get_activity(csvlist):
    activity=pd.DataFrame()
    for i in range(0,len(csvlist)):
        atom_name="P"+str(i).zfill(2)
        activity[atom_name]=pd.read_csv(csvlist[i],sep=" ",index_col=False).Psi
    print(activity.keys())
    return activity
       
def activity_to_occupancy(activity,pdb_in,pdb_out,activity_threshold):
    pdb_out_str=""
    filein=open(pdb_in,"r")
    model=0
    model_end=0
    model_bool=False
    for line in filein:
        if ("MODEL" in line):
            model_bool=True
            model=int(line.split()[1])-1
            pdb_out_str=pdb_out_str+line
            continue
        if ((line.startswith("ATOM") or (line.startswith("HETATM"))) and line.split()[2] in activity.keys() and line.split()[3]=="PRB"):
            if model_bool==False: # some pdb exports don't have MODEL in them
                model=model_end
            if activity[line.split()[2]][model]<activity_threshold:
                continue
            occ_str="{:.2f}".format(round(activity[line.split()[2]][model],2))
            line=(line[0:56]+occ_str+line[60:])
            pdb_out_str=pdb_out_str+line
            continue
        if line.startswith("END"):
            model_end=model_end+1
        pdb_out_str=pdb_out_str+line
    filein.close()

    fileout=open(pdb_out,"w")
    fileout.write(pdb_out_str)
    fileout.close()
    return pdb_out

if __name__=="__main__":

   args=parse()
   print(args)
   activity=get_activity(args.input_csv)
   pdb_out=activity_to_occupancy(activity,args.input_pdb,args.output_pdb,args.activity_threshold)
   
   