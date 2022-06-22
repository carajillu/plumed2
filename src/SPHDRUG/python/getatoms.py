import argparse
import mdtraj

solvent_resn=["HOH","SOL","WAT"]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_gro', nargs="?", help="GMX structure in gro or pdb format",default="traj.gro")
    args = parser.parse_args()
    return args

def process(gro):
    structure=mdtraj.load(gro)
    atomlist=[]
    for atom in structure.topology.atoms:
        if atom.residue.name not in solvent_resn and atom.element!=mdtraj.element.hydrogen:
            atomlist.append(str(atom.serial))
    return atomlist

if __name__=="__main__":
    args=parse()

    atomlist=process(args.input_gro)
    print(",".join(atomlist))