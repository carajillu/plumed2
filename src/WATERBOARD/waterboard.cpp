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
#include "com.h"

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

  vector<AtomNumber> ligand;
  unsigned n_ligand;
  vector<vector<double>> ligand_xyz;
  vector<double> masses_ligand;
  double ligand_total_mass;
  Com ligand_com;

  vector<AtomNumber> water;
  unsigned n_water;
  vector<vector<double>> water_xyz;

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
  masses_ligand=vector<double>(n_ligand,0);
  cout << "Ligand has " << n_ligand << " atoms" << endl;
  atoms.insert(atoms.end(),ligand.begin(),ligand.end());
  water_xyz=vector<vector<double>>(n_water,vector<double>(3,0));

  parseAtomList("WATER",water);
  n_water=water.size();
  cout << "Water has " << n_water << " atoms" << endl;
  atoms.insert(atoms.end(),water.begin(),water.end());
  water_xyz=vector<vector<double>>(n_water,vector<double>(3,0));
  
  n_atoms=atoms.size();
  cout << "Plumed is going to request " << n_atoms << "  atoms" << endl;
  requestAtoms(atoms);

  checkRead();

  cout << "Initialising WATERBOARD derivatives" << endl;
  d_wtb_dx=vector<double>(n_atoms,0);
  d_wtb_dy=vector<double>(n_atoms,0);
  d_wtb_dz=vector<double>(n_atoms,0);
  cout << "     ... deerivatives initialised." << endl;
  
  cout << "---------------------- WATERBOARD initialisation complete ----------------------" << endl;

  ligand_com=Com();
}


// calculator
void Waterboard::calculate() {
  unsigned step=getStep();

  //Initialise
  if (step==0)
  {
    cout << "get ligand masses" << endl;
    for (unsigned j=0; j<n_ligand; j++)
    {
     masses_ligand[j]=getMass(j);
    }
    ligand_com.init(n_ligand,masses_ligand);
  }

  //Get atom coordinates
  #pragma omp parallel for
  for (unsigned j = 0; j < n_atoms; j++)
   {
    if (j<n_ligand)
    {
     ligand_xyz[j][0]=getPosition(j)[0];
     ligand_xyz[j][1]=getPosition(j)[1];
     ligand_xyz[j][2]=getPosition(j)[2];
    }
    else
    {
     water_xyz[j][0]=getPosition(j)[0];
     water_xyz[j][1]=getPosition(j)[1];
     water_xyz[j][2]=getPosition(j)[2];
    }
   }

   //calculate COM
   ligand_com.calculate_com(ligand_xyz); 

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



