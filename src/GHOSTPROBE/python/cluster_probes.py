import mdtraj
import numpy as np
import argparse
import clustering

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_pdb', nargs="?", help="probe coordinates pdb",default="probes.pdb")
    parser.add_argument('-d0','--d0',nargs="?",help="d0 value for laio clustering",type=float,default=0.1)
    parser.add_argument('-delta0','--delta0',nargs="?",help="minimum delta value for a point to be considered a laio cluster center",type=float,default=1.0)
    parser.add_argument('-wmin','--wmin',nargs="?",help="clusters with a diameter lower than wmin are considered false positives",type=float,default=1.0)
    parser.add_argument('-o','--output_pdb', nargs="?", help="output pdb file name",default="cluster.pdb")
    args = parser.parse_args()
    return args

if __name__=="__main__":

    args=parse()
    print(args)
    traj_obj=mdtraj.load_pdb(args.input_pdb)
    rmat=clustering.calc_distance_matrix(traj_obj.xyz[0])
    rhodelta=clustering.laio(rmat,args.d0,args.delta0)
    
    clusters=rhodelta.cluster.unique()
    for cluster in clusters:
        probes_id=rhodelta.point[rhodelta.cluster==cluster]
        top=mdtraj.Topology()
        chain=top.add_chain()
        xyz=traj_obj.xyz[0][probes_id]
        rmax=np.max(clustering.calc_distance_matrix(xyz))
        if (rmax<args.wmin):
            filename="falsepositive_cluster"+str(cluster).zfill(3)+".pdb"
        else:
            filename="cluster"+str(cluster).zfill(3)+".pdb"
        for element in xyz:
            residue=top.add_residue("PRB",chain)
            top.add_atom("PR",mdtraj.element.helium,residue)
        trj=mdtraj.Trajectory(xyz,top)
        trj.save_pdb(filename)