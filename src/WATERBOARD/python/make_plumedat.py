import mdtraj
import argparse

water_resnames=["HOH","SOL","WAT"]
hydrogen_elements=[mdtraj.element.hydrogen,mdtraj.element.virtual_site]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', nargs="?", type=str, help="MD structure file (pdb/gro). MUST HAVE ALL WATERS.",default="structure.pdb")
    parser.add_argument('-l','--ligand', nargs="?", type=str, help="Residue name of the ligand (must be only one).",default="structure.pdb")
    parser.add_argument('-r0','--r0', nargs="?", type=float, help="R0 parameter",default=0.4)
    parser.add_argument('-o','--output', nargs="?", type=str, help="Output file name",default="plumed.dat")
    args = parser.parse_args()
    print(args)
    return args

def get_atoms(z,ligresname):
    water_atoms=[]
    ligand_atoms=[]
    for atom in z.topology.atoms:
       if atom.residue.name==ligresname and atom.element not in hydrogen_elements:
          ligand_atoms.append(atom.serial)
       elif atom.residue.name in water_resnames and atom.element not in hydrogen_elements:
          water_atoms.append(atom.serial)
    return ligand_atoms, water_atoms

def make_plumedat(ligand_atoms,water_atoms,r0,out):
   ligand_atoms = list(map(str, ligand_atoms))
   water_atoms = list(map(str, water_atoms))
   txt="WATERBOARD ...\n"
   txt+="LABEL=w\n"
   txt+="LIGAND="+",".join(ligand_atoms)+"\n"
   txt+="WATER="+",".join(water_atoms)+"\n"
   txt+="R0="+str(r0)+"\n"
   txt+="... WATERBOARD"
   fileout=open(out,"w")
   fileout.write(txt)
   fileout.close()
   return

def main(z,ligresname,r0,output):
    ligand_atoms, water_atoms=get_atoms(z,ligresname)
    make_plumedat(ligand_atoms,water_atoms,r0,output)

if __name__=="__main__":
   args=parse()
   z=mdtraj.load(args.input)
   ligresname=args.ligand
   r0=args.r0
   output=args.output
   main(z,ligresname,r0,output)
