import pandas as pd
import mdtraj 
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pymp
import pickle
import os
import sys

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_gro', nargs="?", help="system coordinates gro",default="system.gro")
    parser.add_argument('-t','--input_traj', nargs="?", help="trajectory file(s)",default=["traj.xtc"])
    parser.add_argument('-s','--selection',  nargs='+',help="atom selection to calculate bonds",default=[None])
    parser.add_argument('-o','--output', nargs="?", help="output csv file",default="hbonds")
    args = parser.parse_args()
    return args

def find_atomid_pair(traj,atomstring_pair):
    atomid_pair=[]
    atomstring_pair=atomstring_pair.split(",")
    for i in range(traj.n_atoms):
        if str(traj.topology.atom(i)) in atomstring_pair:
            atomid_pair.append(i)
    if len(atomid_pair)!=2:
        print(atomid_pair)
        sys.exit("There is something wrong with your selection")
    return atomid_pair


if __name__=="__main__":
    args=parse()
    traj=mdtraj.load(args.input_traj,top=args.input_gro)
    atompairs=[]
    for pair in args.selection:
        atompairs.append(find_atomid_pair(traj,pair))
    
    dist=mdtraj.compute_distances(traj,atompairs).transpose()
    hbonds=pd.DataFrame()
    for i in range(len(args.selection)):
        hbonds[args.selection[i]]=dist[i]
        print(f'{args.selection[i]} = {np.mean(dist[i])} +/- {np.std(dist[i])}')
    hbonds.to_csv(args.output+".csv")
    hbonds.to_pickle(args.output+".pkl")
    