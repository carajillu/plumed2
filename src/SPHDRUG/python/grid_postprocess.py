import argparse
import pandas as pd
import mdtraj 
import sys

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_gro', nargs="?", help="GMX structure in gro or pdb format",default="traj.gro")
    parser.add_argument('-t','--traj', nargs="?", help="GMX trajectory in xtc or trr format",default="traj.xtc")
    parser.add_argument('-g','--grid_xyz', nargs="?", help="Grid in xyz format",default="grid-0.xyz")
    parser.add_argument('-s','--stats', nargs="?", help="Grid stats file",default="grid-0-stats.xyz")
    parser.add_argument('-stride','--stride',type=int, nargs="?", help="Stride of the input files (default=2500)",default=2500)
    parser.add_argument('-o','--output', nargs="?", help="Grid output in xyz format",default="grid-0-out.xyz")

    args = parser.parse_args()
    return args

def parse_grid(grid_name):
    grid_crd=[]
    linum=0
    filein=open(grid_name,"r")
    for line in filein:
        linum=linum+1
        if linum<3:
            continue
        line=line.split()
        crd=[float(line[1])/10,float(line[2])/10,float(line[3])/10]
        grid_crd.append(crd)
    filein.close()
    print(grid_crd)
    return grid_crd

def process_grid(traj,stats,stride):
    centre_crd_frames=[]
    step=0
    newcentre=[0.,0.,0.]
    total_soff=0.
    for i in range(0,len(stats)):
        step_i=int(stats.Step[i]/stride)
        j_index=stats.j_index[i]
        soff=stats.Bsite_centre[i]
        if step_i!=step:
           newcentre[0]/=total_soff
           newcentre[1]/=total_soff
           newcentre[2]/=total_soff
           centre_crd_frames.append(newcentre)
           newcentre=[0.,0.,0.]
           total_soff=0.
           step=step_i
        newcentre[0]+=traj.xyz[step_i][j_index][0]*soff
        newcentre[1]+=traj.xyz[step_i][j_index][1]*soff
        newcentre[2]+=traj.xyz[step_i][j_index][2]*soff
        total_soff+=soff
    
    #Add the last grid snapshot
    newcentre[0]/=total_soff
    newcentre[1]/=total_soff
    newcentre[2]/=total_soff
    centre_crd_frames.append(newcentre)

    return centre_crd_frames

def move_grid(grid_crd,centre_crd_frames):
    centre_0=[0.,0.,0.]
    for point in grid_crd:
        centre_0[0]+=point[0]
        centre_0[1]+=point[1]
        centre_0[2]+=point[2]
    centre_0[0]/=len(grid_crd)
    centre_0[1]/=len(grid_crd)
    centre_0[2]/=len(grid_crd)

    grid_traj=[]

    for frame in centre_crd_frames:
        grid_frame=[]
        for point in grid_crd:
            x=point[0]-centre_0[0]+frame[0]
            y=point[1]-centre_0[1]+frame[1]
            z=point[2]-centre_0[2]+frame[2]
            grid_frame.append([x,y,z])
        grid_traj.append(grid_frame)

    return grid_traj

def print_grid_output(outname,grid_traj):
    fileout=open(outname,"w")
    for grid in grid_traj:
        fileout.write(str(len(grid))+"\n")
        fileout.write("Grid\n")
        for point in grid:
            line="H "+str(round((point[0]*10),5))+" "+str(round((point[1]*10),5))+" "+str(round((point[2]*10),5))+"\n"
            fileout.write(line)
    fileout.close()

if __name__=="__main__":

    args=parse()
    grid_crd=parse_grid(args.grid_xyz)
    traj=mdtraj.load(args.traj,top=args.input_gro)
    stats=pd.read_csv(args.stats,sep=" ")
    centre_crd_frames=process_grid(traj,stats,args.stride)
    grid_traj=move_grid(grid_crd,centre_crd_frames)
    print_grid_output(args.output,grid_traj)
