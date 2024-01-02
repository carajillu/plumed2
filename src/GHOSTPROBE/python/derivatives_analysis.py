import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Process some floats.")
    parser.add_argument('--derivatives', type=str, default="derivatives.csv", help='plumed.dat template')
    parser.add_argument('--nbins', type=int, default=10, help='Number of bins for the histogram')

    args = parser.parse_args()
    return args

def hist2d(x,y,x_str,y_str,nbins,figout):
    g = sns.JointGrid(x=x, y=y)
    hexbin = g.ax_joint.hexbin(x, y, gridsize=nbins, bins="log", cmap='plasma', mincnt=1,vmin=1,vmax=55968)

    # Setting labels
    g.set_axis_labels(x_str, y_str)
    g.ax_joint.set_xlim([-0.3, 0.3])
    #g.ax_joint.set_ylim([-2, 2])
    plt.subplots_adjust(top=1.0)
    plt.xlabel(x_str,fontweight="bold")
    plt.ylabel(y_str,fontweight="bold")
    cbar = plt.colorbar(hexbin, ax=g.ax_joint, orientation='vertical', pad=0.1)
    cbar.set_label('log(count)')
    plt.savefig(figout, dpi=300)
    # Get and print vmin and vmax values
    vmin, vmax = hexbin.get_clim()
    print("vmin:", vmin, "vmax:", vmax)

if __name__=="__main__":
    args=parse_args()
    derivatives=pd.read_csv(args.derivatives,sep=" ")
    hist2d(derivatives.dx,derivatives.correction,"dx","correction",args.nbins,"correction_dx.png")
    hist2d(derivatives.dy,derivatives.correction,"dy","correction",args.nbins,"correction_dy.png")
    hist2d(derivatives.dz,derivatives.correction,"dz","correction",args.nbins,"correction_dz.png")