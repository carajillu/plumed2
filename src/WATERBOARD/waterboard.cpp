/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2023 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "colvar/Colvar.h"
#include "core/ActionRegister.h"

//CV modules
#include "corefunctions.h"

using namespace std;

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR WATERBOARD
/*
This file provides a template for if you want to introduce a new CV.

<!-----You should add a description of your CV here---->

\par Examples

<!---You should put an example of how to use your CV here--->

\plumedfile
# This should be a sample input.
t: WATERBOARD ATOMS=1,2
PRINT ARG=t STRIDE=100 FILE=COLVAR
\endplumedfile
<!---You should reference here the other actions used in this example--->
(see also \ref PRINT)

*/
//+ENDPLUMEDOC

class Waterboard : public Colvar {
  bool nopbc; //because by default it will be yespbc

  vector<AtomNumber> atoms;
  unsigned n_atoms;
  vector<vector<double>> atoms_xyz;

  vector<AtomNumber> ligand;
  unsigned n_ligand;
  vector<double> masses_ligand;
  double ligand_total_mass;
  vector<double> com_ligand;
  vector<double> d_com_dxyz;

  vector<AtomNumber> water;
  unsigned n_water;

  vector<double> rx;
  vector<double> ry;
  vector<double> rz;
  vector<double> r;
  
  vector<double> dr_dx;
  vector<double> dr_dy;
  vector<double> dr_dz;

  double wtb;
  vector<double> d_wtb_dx;
  vector<double> d_wtb_dy;
  vector<double> d_wtb_dz;
public:
  explicit Waterboard(const ActionOptions&);
// active methods:
  void calculate() override;
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(Waterboard,"WATERBOARD")

void Waterboard::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);
  keys.add("atoms","LIGAND","Specify ligand atoms");
  keys.add("atoms","WATER","Specify water atoms");
}

Waterboard::Waterboard(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  nopbc(false)
{
  addValueWithDerivatives(); 
  setNotPeriodic();

  parseFlag("NOPBC",nopbc);
  if(nopbc)
     log.printf("Using PLUMED without periodic boundary conditions\n"); 
  else
     log.printf("Using PLUMED with periodic boundary conditions\n");
  
  cout << "---------------------- Initialising WATERBOARD collective variable ----------------------" << endl;
  parseAtomList("LIGAND",ligand);
  n_ligand=ligand.size();
  cout << "Ligand has " << n_ligand << " atoms" << endl;
  atoms.insert(atoms.end(),ligand.begin(),ligand.end());

  parseAtomList("WATER",water);
  n_water=water.size();
  cout << "Water has " << n_water << " atoms" << endl;
  atoms.insert(atoms.end(),water.begin(),water.end());
  
  n_atoms=atoms.size();
  cout << "Plumed is going to request " << n_atoms << "  atoms" << endl;
  requestAtoms(atoms);

  checkRead();

  // com vectors
  com_ligand=vector<double>(3);
  d_com_dxyz=vector<double>(n_ligand);
  
  // distance vetors
  rx=vector<double>(n_water,0);
  ry=vector<double>(n_water,0);
  rz=vector<double>(n_water,0);
  r=vector<double>(n_water,0);
  
  dr_dx=vector<double>(n_water,0);
  dr_dy=vector<double>(n_water,0);
  dr_dz=vector<double>(n_water,0);

  cout << "Initialising atom coordinate vectors ..." << endl;
  for (unsigned j=0; j<n_atoms; j++)
  {
    atoms_xyz.push_back(vector<double>(3));
  }
  cout << "     ... coordinate vectors initialised." << endl;

  cout << "Initialising WATERBOARD derivatives" << endl;
  d_wtb_dx=vector<double>(n_atoms,0);
  d_wtb_dy=vector<double>(n_atoms,0);
  d_wtb_dz=vector<double>(n_atoms,0);
  cout << "     ... deerivatives initialised." << endl;
  
  cout << "---------------------- WATERBOARD initialisation complete ----------------------" << endl;
}


// calculator
void Waterboard::calculate() {
  unsigned step=getStep();

  //This should be done at setup, but getMass seems not to work before calculate
  if (step==0)
  {
    for (unsigned j = 0; j < n_ligand; j++)
    {
     masses_ligand.push_back(getMass(j));
     d_com_dxyz.push_back(getMass(j));
     ligand_total_mass+=getMass(j);
    }
    for (unsigned j=0;j<n_ligand;j++)
    {
     d_com_dxyz[j]/=ligand_total_mass;
    }
  }

  

  //#pragma omp parallel for
  for (unsigned j = 0; j < n_atoms; j++)
   {
    atoms_xyz[j][0] = getPosition(j)[0];
    atoms_xyz[j][1] = getPosition(j)[1];
    atoms_xyz[j][2] = getPosition(j)[2];
   }
   com_ligand=COREFUNCTIONS::calculate_com(atoms_xyz,n_ligand,masses_ligand,ligand_total_mass);
   cout << "Ligand_com: " << com_ligand[0] << "," << com_ligand[1] << "," << com_ligand[2] << "," << endl;





  setValue(wtb);

  for (unsigned j=0;j<n_atoms;j++)
  {
   //cout << "Assigning derivative " << j << ": " << d_wtb_dx[j] << " " << d_wtb_dy[j] << " " << d_wtb_dz[j] << " " << endl;
   setAtomsDerivatives(j,Vector(d_wtb_dx[j],d_wtb_dy[j],d_wtb_dz[j]));
  }
  //setBoxDerivatives  (-invvalue*Tensor(distance,distance));
}

}
}



