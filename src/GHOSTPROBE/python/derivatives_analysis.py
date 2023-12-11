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

def hist2d(x,y,x_str,y_str,nbins,log):
    g = sns.JointGrid(x=x, y=y)
    hexbin = g.ax_joint.hexbin(x, y, gridsize=nbins, cmap='plasma', bins='log', mincnt=1)

    # Adding marginal histograms
    sns.histplot(x, ax=g.ax_marg_x, kde=False)
    sns.histplot(y, ax=g.ax_marg_y, kde=False, orientation='horizontal')

    # Setting labels
    g.set_axis_labels(x_str, y_str)
    plt.subplots_adjust(top=1.0)
    plt.xlabel(x_str,fontweight="bold")
    plt.ylabel(y_str,fontweight="bold")
    cbar = plt.colorbar(hexbin, ax=g.ax_joint, orientation='vertical', pad=0.1)
    cbar.set_label('log(count)')
    plt.savefig("derivatives.png", dpi=300)

if __name__=="__main__":
    args=parse_args()
    derivatives=pd.read_csv(args.derivatives,sep=" ")
    hist2d(derivatives.dx,derivatives.correction,"dx","correction",args.nbins,False)