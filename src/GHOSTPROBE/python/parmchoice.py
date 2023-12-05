import argparse
import os
import numpy as np
import subprocess
import mdtraj
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Process some floats.")
    parser.add_argument('--plumedat', type=str, default="plumed.dat", help='plumed.dat template')
    parser.add_argument('--xtc', type=str, default="md.xtc", help='MD trajectory file')
    parser.add_argument('--gro', type=str, default="md.gro", help='MD topology file')
    parser.add_argument('--lig', type=str, default="BNZ", help='residue name of the ligand')
    parser.add_argument('--step', type=float, default=0.01, help='Step size')
    parser.add_argument('--nbins', type=int, default=100, help='Number of bins for the histogram')

    args = parser.parse_args()
    return args

# Calculate distances between the ligand (normally benzene) and the closest protein heavy atom
def calc_ligmindist(xtc_file, top_file, lig_name):
    # Load the trajectory
    traj = mdtraj.load_xtc(xtc_file, top=top_file)

    # Identify indices for ligand atoms and non-hydrogen protein atoms
    sel="resname "+lig_name+" and not element H"
    ligand_indices = traj.topology.select(sel)  # Assuming ligand is named 'BEN'
    protein_indices = traj.topology.select('protein and not element H')

    # Check if indices are found
    if not ligand_indices.size or not protein_indices.size:
        raise ValueError("Ligand or protein atoms not found in the trajectory")

    closest_distances = []

    for frame in traj:
        # Calculate center of mass of ligand
        ligand_center = mdtraj.compute_center_of_mass(frame.atom_slice(ligand_indices))[0]

        # Calculate distances from ligand center to each non-H protein atom
        distances = np.linalg.norm(frame.xyz[0, protein_indices] - ligand_center, axis=1)

        # Find the minimum distance
        min_distance = np.min(distances)
        closest_distances.append(min_distance)

    return closest_distances

# Define the modified sQM function
def modified_sQM(m, start, end, max_val):
    if m < start:
        return 0
    elif start <= m <= end:
        normalized_m = (m - start) / (end - start)  # Normalize m to the range [0,1]
        return (3 * normalized_m**4 - 2 * normalized_m**6) * max_val
    else:
        return max_val

#Modify plumed.dat and prepare directory
def plumedat_replace(filename, original_string, replacement_string, new_filename):
    # Read the original file
    with open(filename, 'r') as file:
        file_contents = file.read()

    # Replace the specified string
    modified_contents = file_contents.replace(original_string, replacement_string)

    # Write the modified contents to a new file
    with open(new_filename, 'w') as file:
        file.write(modified_contents)

    print(f"File written: {new_filename}")
    return new_filename

def create_directory(string_arg, float_arg):
    # Creating the directory name by combining string and float
    dir_name = f"{string_arg}_{float_arg}/"

    # Check if the directory already exists
    if not os.path.exists(dir_name):
        # Create the directory if it does not exist
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created.")
    else:
        print(f"Directory '{dir_name}' already exists.")
    return dir_name



def run_command(directory, command):
    # Save the current working directory
    original_directory = os.getcwd()

    try:
        # Change to the target directory
        os.chdir(directory)

        # Execute the command
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Capture the output and errors, if any
        output, error = result.stdout, result.stderr
        print("Output:", output.decode())
        if error:
            print("Error:", error.decode())

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Change back to the original directory
        os.chdir(original_directory)
    return output, error

def min_r_hist(min_r,nbins):   
    #Create histogram of min_r
    n, bins = np.histogram(min_r, bins=nbins)

    # Find the bin center with the maximum number of elements and the first non-zero bin
    max_bin_index = np.argmax(n)
    max_elements = n[max_bin_index]
    center_of_max_bin = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2
    first_non_zero_bin = bins[np.nonzero(n)[0][0]]

    # Generate the range for the modified sQM function
    m_values = np.linspace(bins[0], bins[-1], 300)
    # Calculate the modified sQM values
    modified_sQM_vec = np.vectorize(modified_sQM)
    modified_sQM_values = modified_sQM_vec(m_values, first_non_zero_bin, center_of_max_bin, max_elements)
    #modified_sQM_values = modified_sQM_vec(m_values, 0, center_of_max_bin, max_elements)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(6.4, 4.8))  # Default size for a single plot in a word document

    # Plotting the histogram
    ax1.hist(min_r, bins=args.nbins, color="lightblue", edgecolor="white")

    # Set the titles and labels
    ax1.set_xlabel("Minimum probe-protein distance (nm)", fontweight="bold")
    ax1.set_ylabel("Number of elements",rotation=270,labelpad=15, fontweight="bold")

    # Add the red dotted line indicating the center of the max bin
    ax1.axvline(x=center_of_max_bin, color="red", linestyle="dotted", linewidth=2)
    # Add text annotation
    text_x_position = center_of_max_bin*1.03  # Adjust this as needed for better visibility
    text_y_position = ax1.get_ylim()[1]*0.9  # This will place the text at the top of the y-axis
    ax1.text(text_x_position, text_y_position, "{:.4f}".format(center_of_max_bin), verticalalignment='top', horizontalalignment='right', color='red')

    # Create a second y-axis for the scaled number of elements
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel("Number of elements (scaled)", rotation=270, labelpad=15, fontweight="bold")

    # Set the second y-axis ticks to be scaled by the maximum number of elements
    elements_stride=max_elements/len(ax1.get_yticks())
    ax2.set_yticks(np.arange(0,max_elements+elements_stride,elements_stride))
    ax2.set_yticklabels(["{:.4f}".format(tick / max_elements) for tick in ax2.get_yticks()])

    # Plot the modified sQM function on top of the histogram
    ax1.plot(m_values, modified_sQM_values, color="orange", linewidth=2)

    # Remove grid lines
    ax1.grid(False)
    ax2.grid(False)

    plt.savefig("min_r.png", dpi=300)

    print(f"center of max bin = {center_of_max_bin}")
    return center_of_max_bin

def plot_C(C,center_of_max_bin):
    colors=sns.color_palette("colorblind", len(C.keys()))
    fig, ax1 = plt.subplots(figsize=(6.4, 4.8))  # Default size for a single plot in a word document
    # Set the titles and labels
    ax1.set_xlabel("Minimum probe-protein distance (nm)", fontweight="bold")
    ax1.set_ylabel("C",rotation=0,labelpad=15, fontweight="bold")

    diff_min_r=np.inf
    min_r_plot=np.inf
    i_best=0
    for i in range(1,len(C.keys())):
        data=C[C[C.keys()[i]]==1]
        min_r=min(data.min_r)
        if (abs(min_r-center_of_max_bin) < diff_min_r):
           diff_min_r=abs(min_r-center_of_max_bin)
           deltarmin_plot=C.keys()[i]
           min_r_plot=min_r
           color=colors[i]
           i_best=i

    for i in range(1,len(C.keys())):
        alpha=0.5
        s=1
        if (i==i_best):
            continue
        label="$\Delta C = "+"{:.4f}".format(C.keys()[i])+"$"
        ax1.scatter(C.min_r, C[C.keys()[i]], color=colors[i], s=s, label=label,alpha=alpha)

    alpha=1
    s=10
    label="$\Delta C = "+"{:.4f}".format(C.keys()[i_best])+"$"
    ax1.scatter(C.min_r, C[C.keys()[i_best]], color=colors[i_best], s=s, label=label,alpha=alpha)
    ax1.axvline(x=min_r_plot, color=color, linestyle="dotted", linewidth=2)
    text_x_position = min_r_plot*1.05  # Adjust this as needed for better visibility
    text_y_position = ax1.get_ylim()[1]*0.995  # This will place the text at the top of the y-axis
    ax1.text(text_x_position, text_y_position, "{:.4f}".format(min_r_plot), verticalalignment='top',\
             horizontalalignment='right', color=color, fontweight="bold")
    ax1.grid(False)
    plt.legend()
    plt.savefig("C", dpi=300)

    deltarmin="{:.4f}".format(deltarmin_plot)
    print(f"DELTARMIN = {deltarmin}")
    return deltarmin_plot

def plot_enclosure(filein,nbins):
    data=pd.read_csv(filein,delimiter=" ")
    Pmin=min(data.enclosure)

    # Calculate the histogram data.enclosure
    n, bins = np.histogram(data.enclosure, bins=nbins)

    # Find the bin center with the maximum number of elements and the first non-zero bin
    max_bin_index = np.argmax(n)
    max_elements = n[max_bin_index]
    center_of_max_bin = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2
    deltaP=center_of_max_bin-Pmin

    fig, ax1 = plt.subplots(figsize=(6.4, 4.8))  # Default size for a single plot in a word document
    # Plotting the histogram
    ax1.hist(data["enclosure"], bins=nbins, color="lightblue", edgecolor="white")

    # Set the titles and labels
    ax1.set_xlabel("p", fontweight="bold")
    ax1.set_ylabel("Number of elements",rotation=270,labelpad=15, fontweight="bold")
    #ax1.set_title("Histogram of Minimum probe-protein distance")

    # Add the red dotted line indicating the center of the max bin
    ax1.axvline(x=center_of_max_bin, color="red", linestyle="dotted", linewidth=2)

    # Add text annotation
    text_x_position = center_of_max_bin*1.08  # Adjust this as needed for better visibility
    text_y_position = ax1.get_ylim()[1]*0.9  # This will place the text at the top of the y-axis
    ax1.text(text_x_position, text_y_position, str(round(center_of_max_bin,2)), verticalalignment='top', horizontalalignment='right', color='red')

    ax1.grid(False)
    plt.savefig("p", dpi=300)
    
    deltaP_str="{:.4f}".format(deltaP)
    Pmin_str="{:.4f}".format(Pmin)

    print(f"PMIN = {Pmin_str}, DELTAP = {deltaP_str}")
    return Pmin, deltaP

def plot_activity(data,nbins):
    #Plot activity
    fig, ax1 = plt.subplots(figsize=(6.4, 4.8))  # Default size for a single plot in a word document
    # Plotting the histogram
    ax1.hist(data["activity"], bins=nbins, color="lightblue", edgecolor="white",log=True)

    # Set the titles and labels
    ax1.set_xlabel("activity", fontweight="bold")
    ax1.set_ylabel("Number of elements (log)",rotation=270,labelpad=15, fontweight="bold")

    ax1.grid(False)
    plt.savefig("activity", dpi=300)
    return

if __name__=="__main__":

    args=parse_args()
    mf=args.xtc.split(".")[-1]
    #Get min_r
    if os.path.isfile("min_r.csv"):
        min_r=pd.read_csv("min_r.csv").min_r
    else:
        min_r=calc_ligmindist(args.xtc,args.gro,args.lig)
        z=pd.DataFrame()
        z["min_r"]=min_r
        z.to_csv("min_r.csv")

    center_of_max_bin=min_r_hist(min_r,args.nbins)
    
    #Obtain optmal DELTARMIN AND RMAX
    deltarmin_range=np.arange(center_of_max_bin,max(min_r)+args.step,args.step)
    print(deltarmin_range)
    rmax=max(min_r)
    dir_range=[]
    #make all the directories and run driver
    for deltarmin in deltarmin_range:
        deltarmin_str="{:.4f}".format(deltarmin)
        dir_name=create_directory("plumed",deltarmin_str)
        dir_range.append(dir_name)
        if os.path.isfile(dir_name+"/probe-0-stats.csv"):
            continue
        fileout=dir_name+"/"+args.plumedat
        fileout=plumedat_replace(args.plumedat,"deltarmin",deltarmin_str,fileout)
        fileout=plumedat_replace(fileout,"rmax","{:.4f}".format(max(min_r)),fileout)
        fileout=plumedat_replace(fileout,"pmin","10.0",fileout)
        fileout=plumedat_replace(fileout,"deltap","10.0",fileout)
        cmd=f"plumed driver --mf_{mf} ../{args.xtc} --plumed {args.plumedat}"
        print(cmd)
        output,error=run_command(dir_name,cmd)

    C=pd.DataFrame()
    C["min_r"]=min_r
    diff_min_r=np.inf
    for i in range(0,len(deltarmin_range)):
        filein=dir_range[i]+"probe-0-stats.csv"
        C[deltarmin_range[i]]=pd.read_csv(filein,delimiter=" ").C
    deltarmin_plot=plot_C(C,center_of_max_bin)

    #OBTAIN OPTIMAL PMIN AND DELTAP
    filein=args.plumedat.split(".")[-2]+"_"+"{:.4f}".format(deltarmin_plot)+"/probe-0-stats.csv"
    Pmin,deltaP=plot_enclosure(filein,args.nbins)

    #CALCULATE ACTIVITY
    dir_name=create_directory("plumed_paramopt", 0.0)
    fileout=dir_name+"/"+args.plumedat
    fileout=plumedat_replace(args.plumedat,"deltarmin","{:.4f}".format(deltarmin_plot),fileout)
    fileout=plumedat_replace(fileout,"rmax","{:.4f}".format(max(min_r)),fileout)
    fileout=plumedat_replace(fileout,"pmin","{:.4f}".format(Pmin),fileout)
    fileout=plumedat_replace(fileout,"deltap","{:.4f}".format(deltaP),fileout)
    cmd=f"plumed driver --mf_{mf} ../{args.xtc} --plumed {args.plumedat}"
    output,error=run_command(dir_name,cmd)
    filein=dir_name+"/probe-0-stats.csv"
    data=pd.read_csv(filein,sep=" ")
    plot_activity(data,args.nbins)
    
    

    

    