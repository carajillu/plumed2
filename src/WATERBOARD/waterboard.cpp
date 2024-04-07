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
#include <iostream>
#include <omp.h>

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

  //Parameters
  double r0=0;

  // all atoms
  vector<AtomNumber> atoms;
  unsigned n_atoms;

  //ligand atoms
  vector<AtomNumber> ligand;
  unsigned n_ligand;
  vector<vector<double>> ligand_xyz;
  vector<double> masses_ligand;
  double ligand_total_mass;
  vector<double> com_xyz; //coordinates of the ligand COM, size()=3
  vector<double> dcom_dligand; // derivative of com respect to ligand atom, size()=n_ligand

  //water atoms
  vector<AtomNumber> water;
  unsigned n_water;
  vector<vector<double>> water_xyz;

  //distances between COM and water molecules (COM-water), size()=n_water
  vector<double> rx;
  vector<double> ry;
  vector<double> rz;
  vector<double> r;
  vector<double> dr_dxcom;
  vector<double> dr_dycom;
  vector<double> dr_dzcom;
  //derivatives of the distance with respect to each atom, size()=n_atoms
  vector<double> dr_dx;
  vector<double> dr_dy;
  vector<double> dr_dz;

  //exp(-(r-R0)^2), size()=n_water
  vector<double> er;
  vector<double> der_dr;
  //derivatives of exp(-(r-R0)^2), size()=n_atoms
  vector<double> der_dx;
  vector<double> der_dy;
  vector<double> der_dz;

  //wtb score, size()=n_atoms
  double wtb;
  vector<double> d_wtb_dx;
  vector<double> d_wtb_dy;
  vector<double> d_wtb_dz;
public:
  explicit Waterboard(const ActionOptions&);
// active methods:
  void calculate() override;
  void reset();
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(Waterboard,"WATERBOARD")

void Waterboard::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);
  keys.add("atoms","LIGAND","Specify ligand atoms");
  keys.add("atoms","WATER","Specify water atoms");
  keys.add("optional","R0","Distance COM-water at which the function starts to decrease");
}

Waterboard::Waterboard(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  nopbc(false)
{
  int nthreads;
  #pragma omp parallel 
  {
   nthreads = omp_get_num_threads();
  }
  cout << "---------------------- Available Computing Resources ------------------------------------" << endl;
  cout << "PLUMED initialised with " << nthreads << " OMP threads " << endl;
  addValueWithDerivatives(); 
  setNotPeriodic();

  parseFlag("NOPBC",nopbc);
  if(nopbc)
     log.printf("Using PLUMED without periodic boundary conditions\n"); 
  else
     log.printf("Using PLUMED with periodic boundary conditions\n");
  
  cout << "---------------------- Initialising WATERBOARD collective variable ----------------------" << endl;
  cout << "Parsing parameters..." << endl;
  parse("R0",r0);
  if (!r0)
     r0=0.4;
  cout << "R0 = 0.4 nm" << endl;
  cout << "... Parameters parsed." << endl;

  cout << "Parsing atomlists ..." << endl;
  parseAtomList("LIGAND",ligand);
  n_ligand=ligand.size();
  masses_ligand=vector<double>(n_ligand,0);
  cout << "Ligand has " << n_ligand << " atoms" << endl;
  atoms.insert(atoms.end(),ligand.begin(),ligand.end());
  ligand_xyz=vector<vector<double>>(n_ligand,vector<double>(3,0));
  com_xyz=vector<double>(3,0); //coordinates of the ligand COM, size()=3
  dcom_dligand=vector<double>(n_ligand,0); // derivative of com respect to ligand atom, size()=n_ligand

  parseAtomList("WATER",water);
  n_water=water.size();
  cout << "Water has " << n_water << " atoms" << endl;
  atoms.insert(atoms.end(),water.begin(),water.end());
  water_xyz=vector<vector<double>>(n_water,vector<double>(3,0));

  cout << "... atomlists parsed." << endl;  
  n_atoms=atoms.size();
  cout << "Plumed is going to request " << n_atoms << "  atoms" << endl;
  requestAtoms(atoms);

  checkRead();

 cout << "Initialising WATERBOARD components and their derivatives ..." << endl;

  //distances between COM and water molecules (COM-water), size()=n_water
  rx=vector<double>(n_water,0);
  ry=vector<double>(n_water,0);
  rz=vector<double>(n_water,0);
  r=vector<double>(n_water,0);
  //derivatives of the distance with respect to each atom, size()=n_atoms
  dr_dx=vector<double>(n_atoms,0);
  dr_dy=vector<double>(n_atoms,0);
  dr_dz=vector<double>(n_atoms,0);

  dr_dxcom=vector<double>(n_water,0);
  dr_dycom=vector<double>(n_water,0);
  dr_dzcom=vector<double>(n_water,0);

  //exp(-(r-R0)^2), size()=n_water
  er=vector<double>(n_water,0);
  der_dr=vector<double>(n_water,0);

  d_wtb_dx=vector<double>(n_atoms,0);
  d_wtb_dy=vector<double>(n_atoms,0);
  d_wtb_dz=vector<double>(n_atoms,0);
  cout << "... WATERBOARD components and derivatives initialised." << endl;
  
  cout << "---------------------- WATERBOARD initialisation complete ----------------------" << endl;

}

void Waterboard::reset()
{
  fill(com_xyz.begin(), com_xyz.end(), 0);
  fill(dr_dx.begin(), dr_dx.end(), 0);
  fill(dr_dy.begin(), dr_dy.end(), 0);
  fill(dr_dz.begin(), dr_dz.end(), 0);
  wtb=0;
  fill(d_wtb_dx.begin(), d_wtb_dx.end(), 0);
  fill(d_wtb_dy.begin(), d_wtb_dy.end(), 0);
  fill(d_wtb_dz.begin(), d_wtb_dz.end(), 0);
}
// calculator
void Waterboard::calculate() {
  reset();
  unsigned step=getStep();

  //Get ligand masses and ligand COM derivatives
  if (step==0)
  {
    ligand_total_mass=0;
    //cout << "get ligand masses" << endl;
    for (unsigned l=0; l<n_ligand; l++)
    {
     masses_ligand[l]=1;
     //ligand_total_mass+=getMass(j);
     ligand_total_mass+=masses_ligand[l];
    }
    for (unsigned l=0; l<n_ligand; l++)
    {
      //dcom_dligand[j]=getMass(j)/ligand_total_mass;
      dcom_dligand[l]=masses_ligand[l]/ligand_total_mass;
    }
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
     water_xyz[j-n_ligand][0]=getPosition(j)[0];
     water_xyz[j-n_ligand][1]=getPosition(j)[1];
     water_xyz[j-n_ligand][2]=getPosition(j)[2];
    }
   }

   //Get the coordinates of ligand COM
   for (unsigned l=0; l<ligand.size(); l++)
   {
    com_xyz[0]+=ligand_xyz[l][0]*masses_ligand[l];
    com_xyz[1]+=ligand_xyz[l][1]*masses_ligand[l];
    com_xyz[2]+=ligand_xyz[l][2]*masses_ligand[l];
   }
   com_xyz[0]/=ligand_total_mass;
   com_xyz[1]/=ligand_total_mass;
   com_xyz[2]/=ligand_total_mass;
   //Get distances between ligand COM and waters and their derivatives
   #pragma omp parallel for
   for (unsigned w=0; w<n_water;w++)
   {
    rx[w]=com_xyz[0]-water_xyz[w][0];
    ry[w]=com_xyz[1]-water_xyz[w][1];
    rz[w]=com_xyz[2]-water_xyz[w][2];
    r[w]=sqrt(pow(rx[w],2)+pow(ry[w],2)+pow(rz[w],2));
    dr_dxcom[w]=rx[w]/r[w];
    dr_dycom[w]=ry[w]/r[w];
    dr_dzcom[w]=rz[w]/r[w];
   }

   #pragma omp parallel for
   for (unsigned w=0; w<n_water;w++)
   {
    if (r[w]<=r0)
    {
      er[w]=1;
      der_dr[w]=0;
    }
    else
    {
      er[w]=exp(-pow((r[w]-r0),2));
      der_dr[w]=-2*(r[w]-r0)*er[w];
    }
    #pragma omp critical
    {
     wtb+=er[w];
    }
   }

  #pragma omp parallel for
  for (unsigned j=0; j<n_atoms; j++)
  {
   if (j<n_ligand)
   {
    for (unsigned w=0;w<n_water;w++)
    {
      d_wtb_dx[j]+=der_dr[w]*dr_dxcom[w]*dcom_dligand[j];
      d_wtb_dy[j]+=der_dr[w]*dr_dycom[w]*dcom_dligand[j];
      d_wtb_dz[j]+=der_dr[w]*dr_dzcom[w]*dcom_dligand[j];
    }
   }
   else
   {
    unsigned w=j-n_ligand;
    d_wtb_dx[j]=-der_dr[w]*dr_dxcom[w];
    d_wtb_dy[j]=-der_dr[w]*dr_dycom[w];
    d_wtb_dz[j]=-der_dr[w]*dr_dzcom[w];
   }
   }

  setValue(wtb);
  
  #pragma omp parallel for
  for (unsigned j=0;j<n_atoms;j++)
  {
   //cout << "Assigning derivative " << j << ": " << d_wtb_dx[j] << " " << d_wtb_dy[j] << " " << d_wtb_dz[j] << " " << endl;
   setAtomsDerivatives(j,Vector(d_wtb_dx[j],d_wtb_dy[j],d_wtb_dz[j]));
   //setAtomsDerivatives(j,Vector(dr_dx[j],dr_dy[j],dr_dz[j]));
  }
  //setBoxDerivatives  (-invvalue*Tensor(distance,distance));
}

}
}



