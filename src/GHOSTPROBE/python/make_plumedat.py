import argparse
import pandas as pd
import numpy as np
import mdtraj
import glob
import sys

#default line
defline="""GHOSTPROBE ...
LABEL=d
NPROBES=16 PROBESTRIDE=2500
RMIN=0 DELTARMIN=0.4
RMAX=0.45 DELTARMAX=0.30
CMIN=0 DELTAC=2.5
PMIN=10 DELTAP=10
KPERT=0.1 PERTSTRIDE=1
"""

biasline="RESTRAINT ARG=d AT=1.0 KAPPA=5000 STRIDE=10\n"
printline="PRINT ARG=d FILE=COLVAR STRIDE=2500"


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--input_gro', nargs="?", type=str, help="Reference structure (issued by gmx trjconv)",default="protein.gro")
    parser.add_argument('-s','--selection', nargs="?", type=str, help="Selection of atoms that will go in plumed",default="protein")
    parser.add_argument('--dxclude', nargs="?", type=str, help="Selection of atoms that will NOT experience GHOSTPROBE force",default=None)
    parser.add_argument('-o','--output', nargs="?", type=str, help="Output file",default="plumed.dat")
    args = parser.parse_args()
    print(args)
    return args

def getATOMS(gro,selection,plmd_str="ATOMS"):
    structure=mdtraj.load(gro)
    atomlist=[]
    idlist=structure.topology.select(selection)
    for id in idlist:
            atom=structure.topology.atom(id)
            if (atom.element==mdtraj.element.hydrogen):
                continue
            atomlist.append(str(atom.serial))
    ATOMS_str=plmd_str+"="+",".join(atomlist)
    return ATOMS_str


def build_plumedat(defline,ATOMS_str,DXCLUDE_str,out_file):
    defline=defline+ATOMS_str+"\n"
    if DXCLUDE_str is not None:
        defline=defline+DXCLUDE_str+"\n"
    defline=defline+"... GHOSTPROBE\n"
    fileout=open(out_file,"w")
    fileout.write(defline)
    fileout.write(biasline)
    fileout.write(printline)
    fileout.close()
    

if __name__=="__main__":

    args=parse()
    atomlist=getATOMS(args.input_gro,args.selection)
    if args.dxclude is None:
        dxcludelist=None
    else:
        dxselection=args.selection+" and "+args.dxclude
        dxcludelist=getATOMS(args.input_gro,dxselection,"DXCLUDE")
    
    build_plumedat(defline,atomlist,dxcludelist,args.output)
    
    

    

