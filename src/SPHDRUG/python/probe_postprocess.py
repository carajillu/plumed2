import argparse
import pandas as pd
import mdtraj 
import sys

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_gro', nargs="?", help="GMX structure in gro or pdb format",default="traj.gro")
    parser.add_argument('-t','--traj', nargs="?", help="GMX trajectory in xtc or trr format",default="traj.xtc")
    parser.add_argument('-p','--probe_xyz', nargs="?", help="Probe in xyz format",default="probe-0.xyz")
    parser.add_argument('-m','--movement', nargs="?", help="Probe movement file",default="probe-0-movement.xyz")
    parser.add_argument('-s','--stride',type=int, nargs="?", help="Stride of the input files (default=2500)",default=2500)
    parser.add_argument('-o','--output', nargs="?", help="Probe output in xyz format",default="probe-0-out.xyz")

    args = parser.parse_args()
    return args

def parse_probe(probe_name):
    probe_crd=[]
    linum=0
    filein=open(probe_name,"r")
    for line in filein:
        linum=linum+1
        if linum<3:
            continue
        line=line.split()
        crd=[float(line[1])/10,float(line[2])/10,float(line[3])/10]
        probe_crd.append(crd)
    filein.close()
    print(probe_crd)
    return probe_crd

def process_probe(traj,movement,stride):
    centre_crd_frames=[]
    step=0
    newcentre=[0.,0.,0.]
    total_soff=0.
    for i in range(0,len(movement)):
        step_i=int(movement.Step[i]/stride)
        j_index=movement.j_index[i]
        soff=movement.Soff_r[i]
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
    
    #Add the last probe snapshot
    newcentre[0]/=total_soff
    newcentre[1]/=total_soff
    newcentre[2]/=total_soff
    centre_crd_frames.append(newcentre)

    return centre_crd_frames

def move_probe(probe_crd,centre_crd_frames):
    centre_0=[0.,0.,0.]
    for point in probe_crd:
        centre_0[0]+=point[0]
        centre_0[1]+=point[1]
        centre_0[2]+=point[2]
    centre_0[0]/=len(probe_crd)
    centre_0[1]/=len(probe_crd)
    centre_0[2]/=len(probe_crd)

    probe_traj=[]

    for frame in centre_crd_frames:
        probe_frame=[]
        for point in probe_crd:
            x=point[0]-centre_0[0]+frame[0]
            y=point[1]-centre_0[1]+frame[1]
            z=point[2]-centre_0[2]+frame[2]
            probe_frame.append([x,y,z])
        probe_traj.append(probe_frame)

    return probe_traj

def print_probe_output(outname,probe_traj):
    fileout=open(outname,"w")
    for probe in probe_traj:
        fileout.write(str(len(probe))+"\n")
        fileout.write("Probe\n")
        for point in probe:
            line="H "+str(round((point[0]*10),5))+" "+str(round((point[1]*10),5))+" "+str(round((point[2]*10),5))+"\n"
            fileout.write(line)
    fileout.close()

if __name__=="__main__":

    args=parse()
    probe_crd=parse_probe(args.probe_xyz)
    traj=mdtraj.load(args.traj,top=args.input_gro)
    movement=pd.read_csv(args.movement,sep=" ")
    centre_crd_frames=process_probe(traj,movement,args.stride)
    probe_traj=move_probe(probe_crd,centre_crd_frames)
    print_probe_output(args.output,probe_traj)
