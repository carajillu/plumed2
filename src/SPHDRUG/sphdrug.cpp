/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2019 The plumed team
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
#include "colvar/ActionRegister.h"

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <armadillo>
#include <omp.h>

#define max_atoms 10000

//CV modules

using namespace std;
using  namespace std::chrono;

namespace PLMD {
namespace colvar {

/*+PLUMEDOC COLVAR TEMPLATE
Add CV info
+ENDPLUMEDOC*/

class Sphdrug : public Colvar {
  // Execution control variables
  int nthreads; //number of available OMP threads
  int ndev; // number of available OMP accelerators
  bool performance; // print execution time
  // MD control variables
  bool pbc;
  // CV control variables
  bool nodxfix;
  bool noupdate;
  // Variables necessary to check results
  bool target;
  // Parameters
  double rprobe; // radius of each spherical probe
  double mind_slope; //slope of the mind linear implementation
  double mind_intercept; //intercept of the mind linear implementation
  double CCmin; // mind below which an atom is considered to be clashing with the probe 
  double CCmax; // distance above which an atom is considered to be too far away from the probe*
  double deltaCC; // interval over which contact terms are turned on and off
  double Pmax; // number of atoms surrounding the probe for it to be considered completely packed
  double Dmin; // packing factor below which depth term equals 0
  double deltaD; // interval over which depth term turns from 0 to 1
  // Set up of CV
  vector<PLMD::AtomNumber> atoms; // indices of atoms supplied to the CV (starts at 1)
  unsigned n_atoms; // number of atoms supplied to the CV
  double atoms_x[max_atoms];
  double atoms_y[max_atoms];
  double atoms_z[max_atoms];

  int nprobes; // number of spherical probes to use

  // Output control variables
  unsigned probestride; // stride to print information for post-processing the probe coordinates 

  // Calculation of CV and its derivatives
  double sphdrug;
  vector<double> d_Sphdrug_dx;
  vector<double> d_Sphdrug_dy;
  vector<double> d_Sphdrug_dz;
  
  //Correction of derivatives
  double sum_d_dx;
  double sum_d_dy;
  double sum_d_dz;
  double sum_t_dx;
  double sum_t_dy;
  double sum_t_dz;

  arma::vec L;
  unsigned nrows;
  unsigned ncols;
  arma::mat A;
  arma::mat Aplus;
  arma::vec P;
  vector<double> sum_P;
  vector<double> sum_rcrossP;

public:
  explicit Sphdrug(const ActionOptions&);
// active methods:
  void calculate() override;
  void reset();
  void correct_derivatives();
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(Sphdrug,"SPHDRUG")

void Sphdrug::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);
  keys.addFlag("DEBUG",false,"Running in debug mode");
  keys.addFlag("NOUPDATE",false,"skip probe update");
  keys.addFlag("NODXFIX",false,"skip derivative correction");
  keys.addFlag("PERFORMANCE",false,"measure execution time");
  keys.add("atoms","ATOMS","Atoms to include in druggability calculations (start at 1)");
  keys.add("optional","NPROBES","Number of probes to use");
  keys.add("optional","RPROBE","Radius of every probe in nm");
  keys.add("optional","PROBESTRIDE","Radius of every probe in nm");
  keys.add("optional","CCMIN","Radius of every probe in nm");
  keys.add("optional","CCMAX","Radius of every probe in nm");
  keys.add("optional","DELTACC","Radius of every probe in nm");
  keys.add("optional","MINDSLOPE","Radius of every probe in nm");
  keys.add("optional","MINDINTERCEPT","Radius of every probe in nm");
  //keys.add("deltaCC","DELTACC","Radius of every probe in nm");
}

Sphdrug::Sphdrug(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  noupdate(false),
  nodxfix(false),
  performance(false),
  target(false)
{
  /*
  Initialising openMP threads. 
  This does not seem to be affected by the environment variable $PLUMED_NUM_THREADS
  */
  #pragma omp parallel
     nthreads=omp_get_num_threads();
  ndev=omp_get_num_devices();
  cout << "Sphdrug initialised with " << nthreads << " OMP threads " << endl;
  cout << "and " << ndev << " OMP compatible accelerators (not currently used)" << endl;
  
  addValueWithDerivatives(); 
  setNotPeriodic();

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;

  parseFlag("NOUPDATE",noupdate);
  parseFlag("NODXFIX",nodxfix);
  parseFlag("PERFORMANCE",performance);

  parseAtomList("ATOMS",atoms);
  requestAtoms(atoms);
  n_atoms=atoms.size();

  cout << "--------- Initialising Sphdrug Collective Variable -----------" << endl;
  parse("NPROBES",nprobes);
  if (!nprobes) nprobes=1;
  cout << "Using " << nprobes << " spherical probe(s) ";

  parse("RPROBE",rprobe);
  if (!rprobe) rprobe=0.3;
  cout << "of radius equal to " << rprobe << " nm." << endl;

  parse("PROBESTRIDE",probestride);
  if (!probestride) probestride=1;
  cout << "Information to post-process probe coordinates will be printed every " << probestride << " steps" << endl << endl;

  // PARAMETERS NEEDED TO CALCULATE THE CV
  parse("CCMIN",CCmin);
  if (!CCmin) CCmin=0.2;
  cout << "CCmin = " << CCmin << endl;

  parse("CCMAX",CCmax);
  if (!CCmax) CCmax=0.5;
  cout << "CCmax = " << CCmax << endl;

  parse("DELTACC",deltaCC);
  if (!deltaCC) deltaCC=0.05;
  cout << "deltaCC = " << deltaCC << endl;

  parse("MINDSLOPE",mind_slope);
  if (!mind_slope) mind_slope=1.227666; //obtained from generating 10000 random points in VHL's crystal structure
  cout << "MINDSLOPE = " << mind_slope << endl;

  parse("MINDINTERCEPT",mind_intercept);
  if (!mind_intercept) mind_intercept=-0.089870; //obtained from generating 10000 random points in VHL's crystal structure
  cout << "MINDINTERCEPT = " << mind_intercept << endl;


  checkRead();

  cout << "Initialisng Sphdrug and its derivatives" << endl;
  sphdrug=0;
  double d_Sphdrug_dx[max_atoms];
  double d_Sphdrug_dy[max_atoms];
  double d_Sphdrug_dz[max_atoms];

  if (!nodxfix)
  {
  cout << "Initialisng correction of Sphdrug derivatives" << endl;

  //L=vector<double>(6,0); //sums of derivatives and sums torques in each direction
  nrows=6;
  ncols=3*n_atoms;
  A=arma::mat(nrows,ncols);
  Aplus=arma::mat(nrows,ncols);
  L=arma::vec(nrows);
  P=arma::vec(ncols);
  sum_P=vector<double>(3,0);
  sum_rcrossP=vector<double>(3,0);
  }
  else
  {
    cout << "Sphdrug derivatives are not going to be corrected"<<endl;
    cout << "Use the NODXFIX flag with care, as this means that"<<endl;
    cout << "the sum of forces in the system will not be zero"<<endl;
  }

  cout << "--------- Initialisation complete -----------" << endl;
}

// reset Sphdrug and derivatives to 0
void Sphdrug::reset()
{  
  sphdrug=0;
  fill(d_Sphdrug_dx.begin(),d_Sphdrug_dx.end(),0);
  fill(d_Sphdrug_dy.begin(),d_Sphdrug_dy.end(),0);
  fill(d_Sphdrug_dz.begin(),d_Sphdrug_dz.end(),0);

  sum_d_dx=0;
  sum_d_dy=0;
  sum_d_dz=0;
  sum_t_dx=0;
  sum_t_dy=0;
  sum_t_dz=0;
  fill(L.begin(),L.end(),0);
  fill(P.begin(),P.end(),0);
  fill(sum_P.begin(),sum_P.end(),0);
  fill(sum_rcrossP.begin(),sum_rcrossP.end(),0);
}

void Sphdrug::correct_derivatives()
{
  //auto point0=high_resolution_clock::now();
  //step 0: calculate sums of derivatives and sums of torques in each direction
  for (unsigned j=0;j<n_atoms;j++)
  {
   sum_d_dx+=d_Sphdrug_dx[j];
   sum_d_dy+=d_Sphdrug_dy[j];
   sum_d_dz+=d_Sphdrug_dz[j];

   sum_t_dx+=atoms_y[j]*d_Sphdrug_dz[j]-atoms_z[j]*d_Sphdrug_dy[j];
   sum_t_dy+=atoms_z[j]*d_Sphdrug_dx[j]-atoms_x[j]*d_Sphdrug_dz[j];
   sum_t_dz+=atoms_x[j]*d_Sphdrug_dy[j]-atoms_y[j]*d_Sphdrug_dx[j];
  }
  L[0]=-sum_d_dx;
  L[1]=-sum_d_dy;
  L[2]=-sum_d_dz;
  L[3]=-sum_t_dx;
  L[4]=-sum_t_dy;
  L[5]=-sum_t_dz;
  
  //only for debugging
  //cout << "Before: " << L[0] << " " << L[1] << " " << L[2] << " " << L[3] << " "<< L[4] << " "<< L[5] << endl;
  
  //step2 jedi.cpp
  //auto point1=high_resolution_clock::now();
  #pragma omp parallel for
  for (unsigned j=0; j<ncols; j++)
  {
    if (j<n_atoms)
    {
     A.row(0).col(j)=1.0;
     A.row(1).col(j)=0.0;
     A.row(2).col(j)=0.0;
     A.row(3).col(j)=0.0;
     A.row(4).col(j)=atoms_z[j];
     A.row(5).col(j)=-atoms_y[j];
    }
    else if (j<2*n_atoms)
    {
     A.row(0).col(j)=0.0;
     A.row(1).col(j)=1.0;
     A.row(2).col(j)=0.0;
     A.row(3).col(j)=atoms_z[j-n_atoms];
     A.row(4).col(j)=0.0;
     A.row(5).col(j)=atoms_x[j-n_atoms];
    }
    else
    {
     A.row(0).col(j)=0.0;
     A.row(1).col(j)=0.0;
     A.row(2).col(j)=1.0;
     A.row(3).col(j)=atoms_y[j-2*n_atoms];
     A.row(4).col(j)=-atoms_x[j-2*n_atoms];
     A.row(5).col(j)=0.0;
    }
  }

  //auto point2=high_resolution_clock::now();
  //step3 jedi.cpp
  Aplus=arma::pinv(A);

  
  //auto point3=high_resolution_clock::now();
  //step4 jedi.cpp
  P=Aplus*L;

  //auto point4=high_resolution_clock::now();
  for (unsigned j=0; j<n_atoms;j++)
  {
    sum_P[0]+=P[j+0*n_atoms];
    sum_P[1]+=P[j+1*n_atoms];
    sum_P[2]+=P[j+2*n_atoms];

    sum_rcrossP[0]+=atoms_y[j]*P[j+2*n_atoms]-atoms_z[j]*P[j+1*n_atoms];
    sum_rcrossP[0]+=atoms_z[j]*P[j+0*n_atoms]-atoms_x[j]*P[j+2*n_atoms];
    sum_rcrossP[0]+=atoms_x[j]*P[j+1*n_atoms]-atoms_y[j]*P[j+0*n_atoms];
  }

  //step5 jedi.cpp
  for (unsigned j=0; j<n_atoms;j++)
  {
   d_Sphdrug_dx[j]+=P[j+0*n_atoms];
   d_Sphdrug_dy[j]+=P[j+1*n_atoms];
   d_Sphdrug_dz[j]+=P[j+2*n_atoms];
  }
  //auto point5=high_resolution_clock::now();
  
  //Only for debugging
  /*
  sum_d_dx=0;
  sum_d_dy=0;
  sum_d_dz=0;
  sum_t_dx=0;
  sum_t_dy=0;
  sum_t_dz=0;
  
  for (unsigned j=0;j<n_atoms;j++)
  {
   sum_d_dx+=d_Sphdrug_dx[j];
   sum_d_dy+=d_Sphdrug_dy[j];
   sum_d_dz+=d_Sphdrug_dz[j];

   sum_t_dx+=atom_crd[j][1]*d_Sphdrug_dz[j]-atom_crd[j][2]*d_Sphdrug_dy[j];
   sum_t_dy+=atom_crd[j][2]*d_Sphdrug_dx[j]-atom_crd[j][0]*d_Sphdrug_dz[j];
   sum_t_dz+=atom_crd[j][0]*d_Sphdrug_dy[j]-atom_crd[j][1]*d_Sphdrug_dx[j];
  }
  L[0]=-sum_d_dx;
  L[1]=-sum_d_dy;
  L[2]=-sum_d_dz;
  L[3]=-sum_t_dx;
  L[4]=-sum_t_dy;
  L[5]=-sum_t_dz;
  
  cout << "After: " << L[0] << " " << L[1] << " " << L[2] << " " << L[3] << " "<< L[4] << " "<< L[5] << endl;
  
  for (unsigned k=0; k<P.size();k++)
  {
    cout << P[k] << endl;
  }
  */
  /*
  auto duration0 = duration_cast<microseconds>(point1 - point0);
  auto duration1 = duration_cast<microseconds>(point2 - point1);
  auto duration2 = duration_cast<microseconds>(point3 - point2);
  auto duration3 = duration_cast<microseconds>(point4 - point3);
  auto duration4 = duration_cast<microseconds>(point5 - point4);
  auto duration5 = duration_cast<microseconds>(point5 - point0);
  cout << "Duration L = " << duration0.count() << " microseconds" << endl;
  cout << "Duration A = " << duration1.count() << " microseconds" << endl;
  cout << "Duration Aplus = " << duration2.count() << " microseconds" << endl;
  cout << "Duration P = " << duration3.count() << " microseconds" << endl;
  cout << "Duration Correction = " << duration4.count() << " microseconds" << endl;
  cout << "Duration Total = " << duration5.count() << " microseconds" << endl;
  
  exit(0);
  */
}


// calculator
void Sphdrug::calculate() {
  auto start_psi=high_resolution_clock::now();
  if (pbc) makeWhole();
  reset();
  auto end_psi=high_resolution_clock::now();
  int exec_time=duration_cast<microseconds>(end_psi-start_psi).count();
  cout << "Executed in " << exec_time << " microseconds." << endl;

}//close calculate
}//close colvar
}//close plmd


