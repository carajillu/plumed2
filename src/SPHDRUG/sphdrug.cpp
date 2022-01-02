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

//CV modules
#include "grid.h"

using namespace std;
using  namespace std::chrono;

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR TEMPLATE
/*
This file provides a template for if you want to introduce a new CV.

<!-----You should add a description of your CV here---->

\par Examples

<!---You should put an example of how to use your CV here--->

\plumedfile
# This should be a sample input.
t: TEMPLATE ATOMS=1,2
PRINT ARG=t STRIDE=100 FILE=COLVAR
\endplumedfile
<!---You should reference here the other actions used in this example--->
(see also \ref PRINT)

*/
//+ENDPLUMEDOC

class Psidrug : public Colvar {
  double elapsed_psi=0;
  double elapsed_dfix=0;
  // SETUP
  int nthreads;
  int ndev;
  bool pbc;
  bool debug;
  bool noupdate;
  bool nodxfix;
  bool performance;
  vector<AtomNumber> atoms;
  bool target;
  vector<AtomNumber> target_atoms;
  vector<unsigned> target_atoms_j;
  string grid_file;
  unsigned gridstride;//frequency to print grid
  unsigned taboostride;
  unsigned ngrid=0;
  int nkernelsgrid=0;
  double rgrid=0;
  double rsite=0;
  double rcentre=0;
  double spacing=0;
  unsigned n_atoms;
  vector<grid> grids;
  //PARAMETERS
  double CCmin=0;
  double deltaCC=0;
  double CCmax=0;
  double mind_slope=0;
  double mind_intercept=0;
  //CV
  double PsiDrug;
  vector<double> d_PsiDrug_dx;
  vector<double> d_PsiDrug_dy;
  vector<double> d_PsiDrug_dz;
  int numstep=0;
  vector<Vector> atom_crd; //atom coordinates
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
  explicit Psidrug(const ActionOptions&);
// active methods:
  void calculate() override;
  void reset();
  void correct_derivatives();
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(Psidrug,"PSIDRUG")

void Psidrug::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);
  keys.addFlag("DEBUG",false,"Running in debug mode");
  keys.addFlag("NOUPDATE",false,"Don't update the grid (for derivatives debugging)");
  keys.addFlag("NODXFIX",false,"Don't correct the derivatives");
  keys.addFlag("PERFORMANCE",false,"Print a performance benchmark for every grid and for the whole code");
  keys.add("atoms","ATOMS","Atoms to include in druggability calculations (start at 1)");
  keys.add("atoms","TARGET_ATOMS","Atoms in the target pocket. Need to be among the atoms included in ATOMS");
  keys.add("optional","NGRID","Number of quasi-spherical grids to place in the system (default = 1)");
  keys.add("optional","RGRID","Radius of the quasi-spherical grids that will be placed in the system (default = 0.3 nm)");
  keys.add("optional","SPACING","Space between adjacent grid points (default = 0.1 nm)");
  keys.add("optional","GRIDSTRIDE","Frequence in steps to print grid coordinates");
  keys.add("optional","TABOOSTRIDE","Frequence in steps to apply penalties to already visited places (default = 0 (don't do it))");
  //keys.add("optional","RSITE","Allowed maximum distance from the edge of the grid for any given atom to be taken into account (default = 0.45 nm)");
  keys.add("optional","RCOFF","The influence of an atom in the grid update will decrease to from 1 to 0 from RGRID+RSITE-RCOFF to RGRID+RSITE (default = 0.05 nm)");
  keys.add("optional","CCMIN","Distance at and below which we consider that a grid point is clashing with an atom (default = 0.2 nm)");
  keys.add("optional","DELTACC","Distance interval over which the clash gridpoint-atom is turned off (default = 0.05)");
  keys.add("optional","CCmax","Distance at and below which we consider that a grid point sees an atom (default = 0.4 nm)");
  keys.add("optional","MINDSLOPE","Slope of the linear correction for mindist");
  keys.add("optional","MINDINTERCEPT","Intercept of the linear correction for mindist");
  keys.add("optional","GRID_FILE","XYZ file containing a sample grid. Used for debug purposes.");

}

Psidrug::Psidrug(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  debug(false),
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
  cout << "Psidrug initialised with " << nthreads << " OMP threads " << endl;
  cout << "and " << ndev << " OMP compatible accelerators (not currently used)" << endl;
  
  addValueWithDerivatives(); 
  setNotPeriodic();

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;

  parseFlag("DEBUG",debug);
  parse("GRID_FILE",grid_file);
  if (debug)
  {
     log.printf("RUNNING IN DEBUG MODE\n");
     if(grid_file=="")
        {
          log.printf("Please specify a grid xyz file if you are running in debug mode");
          exit(0);
        }
  }

  parseFlag("NOUPDATE",noupdate);
  parseFlag("NODXFIX",nodxfix);
  parseFlag("PERFORMANCE",performance);

  parseAtomList("ATOMS",atoms);
  requestAtoms(atoms);
  n_atoms=atoms.size();
  atom_crd.reserve(n_atoms);

  parseAtomList("TARGET_ATOMS",target_atoms);
  if(target_atoms.size()>0) 
  {
  target=true;
  for (unsigned j=0; j<n_atoms; j++)
   {
    for (unsigned k=0; k<target_atoms.size();k++)
    {
      if (atoms[j].serial()==target_atoms[k].serial())
      {
        target_atoms_j.push_back(j);
        continue;
      }
    }
   }
  }


  cout << "--------- Initialising Psidrug Collective Variable -----------" << endl;

  parse("NGRID",ngrid);
  if (!ngrid) ngrid=1;
  cout << "Using " << ngrid << " quasi-spherical grid(s)" << endl;
  if (ngrid%nthreads!=0 and nthreads%ngrid!=0)
  {
  cout << "Error: the number of grids is not a multiple of the number or threads, nor the other way around. Exiting. "<<endl;
  exit(0);
  }

  parse("RGRID",rgrid);
  if (!rgrid) rgrid=0.3;
  cout << " of radius equal to " << rgrid << " nm." << endl << endl;

  parse("SPACING",spacing);
  if (!spacing) spacing=0.1;
  cout << "Spacing between adjacent grid points is equal to " << spacing << " nm." << endl << endl;

  parse("GRIDSTRIDE",gridstride);
  if (!gridstride) gridstride=1;
  cout << "Information to post-process grid coordinates will be printed every " << gridstride << " steps" << endl << endl;

  parse("TABOOSTRIDE",taboostride);
  if (!taboostride) 
  {
  taboostride=0;
  cout << "No taboo search will be performed" << endl;
  }
  else
  {
  cout << "Taboo search will be updated every " << taboostride << " steps" << endl << endl;
  }

  // PARAMETERS NEEDED TO CALCULATE THE CV
  parse("CCMIN",CCmin);
  if (!CCmin) CCmin=0.2;
  cout << "CCmin = " << CCmin << endl;

  parse("DELTACC",deltaCC);
  if (!deltaCC) deltaCC=0.05;
  cout << "deltaCC = " << deltaCC << endl;

  parse("CCmax",CCmax);
  if (!CCmax) CCmax=0.5;
  cout << "CCmax = " << CCmax << endl;

  parse("MINDSLOPE",mind_slope);
  if (!mind_slope) mind_slope=1.227666; //obtained from generating 10000 random points in VHL's crystal structure
  cout << "MINDSLOPE = " << mind_slope << endl;

  parse("MINDINTERCEPT",mind_intercept);
  if (!mind_intercept) mind_intercept=-0.089870; //obtained from generating 10000 random points in VHL's crystal structure
  cout << "MINDINTERCEPT = " << mind_intercept << endl;


  // PARAMETERS NEEDED TO APPLY CUTOFFS

  //calculate rcentre so that we can build bsite_bin by measuring from the centre of the grid
  rsite=CCmax+deltaCC;
  cout << "Atoms further away than " << rsite << " from a given grid point will not contribute to its activity" << endl;
  rcentre=rsite+rgrid;
  cout << "Atoms further away than " << rcentre << " from the center of the grid will not contribute to score" << endl;


  checkRead();
  cout << "Initialising grids and kernels..." << endl;
  int total_kernels=0;
  for (unsigned i=0; i<ngrid; i++)
  {
   nkernelsgrid=0;
   grids.push_back(grid(n_atoms,rgrid,spacing,rsite,gridstride,target_atoms_j));
   if (debug)
    {
     string grd=grid_file;
     if (ngrid>1) grd=grid_file+"."+to_string(i);
     cout << "   Reading grid "<< i << ": "<< grd << endl;
     grids[i].grid_read(grd);
    }
   else
    {
      cout << " Initialising grid "<< i << endl;
      grids[i].grid_setup();
    }
   
   int nkernel=0;
   if (ngrid>=nthreads)
      nkernel=1;
   else
   {
     nkernel=nthreads/ngrid;
   }
   for (unsigned k=0;k<nkernel;k++)
   {
   cout << "   Initialising kernel " << k << " on grid " << i << endl;
   grids[i].kernels.push_back(kernel(n_atoms,rsite,CCmin,CCmax,deltaCC,mind_slope,mind_intercept));
   total_kernels++;
   nkernelsgrid++;
   }
   
   //Assign grid points to each created kernel
   int kernel_id=0;
   for (unsigned k=0;k<grids[i].size_grid;k++)
   {
    if (kernel_id==nkernelsgrid) kernel_id=0;
    grids[i].kernels[kernel_id].kernel_points.push_back(k);
    kernel_id++;
   }
  }

  for (unsigned k=0; k<nkernelsgrid;k++)
  {
    cout << "Kernel " << k << " Has points: ";
    for (unsigned i=0; i<grids[0].kernels[k].kernel_points.size();i++)
    {
      cout << grids[0].kernels[k].kernel_points[i] << " ";
    }
    cout << endl;
  }
  //exit();

  cout << "Each of the " << ngrid << " grids has " << nkernelsgrid << " kernels." << endl;
  cout << "A total of " << total_kernels << " have been initialised"<< endl;
  cout << "...Grids and kernels initialised" << endl<<endl;

  cout << "Initialisng Psidrug and its derivatives" << endl;
  PsiDrug=0;
  d_PsiDrug_dx=vector<double>(n_atoms,0);
  d_PsiDrug_dy=vector<double>(n_atoms,0);
  d_PsiDrug_dz=vector<double>(n_atoms,0);

  if (!nodxfix)
  {
  cout << "Initialisng correction of Psidrug derivatives" << endl;

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
    cout << "Psidrug derivatives are not going to be corrected"<<endl;
    cout << "Use the NODXFIX flag with care, as this means that"<<endl;
    cout << "the sum of forces in the system will not be zero"<<endl;
  }

  cout << "--------- Initialisation complete -----------" << endl;
}

// reset psidrug and derivatives to 0
void Psidrug::reset()
{  
  PsiDrug=0;
  fill(d_PsiDrug_dx.begin(),d_PsiDrug_dx.end(),0);
  fill(d_PsiDrug_dy.begin(),d_PsiDrug_dy.end(),0);
  fill(d_PsiDrug_dz.begin(),d_PsiDrug_dz.end(),0);

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

void Psidrug::correct_derivatives()
{
  //auto point0=high_resolution_clock::now();
  //step 0: calculate sums of derivatives and sums of torques in each direction
  for (unsigned j=0;j<n_atoms;j++)
  {
   sum_d_dx+=d_PsiDrug_dx[j];
   sum_d_dy+=d_PsiDrug_dy[j];
   sum_d_dz+=d_PsiDrug_dz[j];

   sum_t_dx+=atom_crd[j][1]*d_PsiDrug_dz[j]-atom_crd[j][2]*d_PsiDrug_dy[j];
   sum_t_dy+=atom_crd[j][2]*d_PsiDrug_dx[j]-atom_crd[j][0]*d_PsiDrug_dz[j];
   sum_t_dz+=atom_crd[j][0]*d_PsiDrug_dy[j]-atom_crd[j][1]*d_PsiDrug_dx[j];
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
     A.row(4).col(j)=atom_crd[j][2];
     A.row(5).col(j)=-atom_crd[j][1];
    }
    else if (j<2*n_atoms)
    {
     A.row(0).col(j)=0.0;
     A.row(1).col(j)=1.0;
     A.row(2).col(j)=0.0;
     A.row(3).col(j)=-atom_crd[j-n_atoms][2];
     A.row(4).col(j)=0.0;
     A.row(5).col(j)=atom_crd[j-n_atoms][0];
    }
    else
    {
     A.row(0).col(j)=0.0;
     A.row(1).col(j)=0.0;
     A.row(2).col(j)=1.0;
     A.row(3).col(j)=atom_crd[j-2*n_atoms][1];
     A.row(4).col(j)=-atom_crd[j-2*n_atoms][0];
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

    sum_rcrossP[0]+=atom_crd[j][1]*P[j+2*n_atoms]-atom_crd[j][2]*P[j+1*n_atoms];
    sum_rcrossP[0]+=atom_crd[j][2]*P[j+0*n_atoms]-atom_crd[j][0]*P[j+2*n_atoms];
    sum_rcrossP[0]+=atom_crd[j][0]*P[j+1*n_atoms]-atom_crd[j][1]*P[j+0*n_atoms];
  }

  //step5 jedi.cpp
  for (unsigned j=0; j<n_atoms;j++)
  {
   d_PsiDrug_dx[j]+=P[j+0*n_atoms];
   d_PsiDrug_dy[j]+=P[j+1*n_atoms];
   d_PsiDrug_dz[j]+=P[j+2*n_atoms];
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
   sum_d_dx+=d_PsiDrug_dx[j];
   sum_d_dy+=d_PsiDrug_dy[j];
   sum_d_dz+=d_PsiDrug_dz[j];

   sum_t_dx+=atom_crd[j][1]*d_PsiDrug_dz[j]-atom_crd[j][2]*d_PsiDrug_dy[j];
   sum_t_dy+=atom_crd[j][2]*d_PsiDrug_dx[j]-atom_crd[j][0]*d_PsiDrug_dz[j];
   sum_t_dz+=atom_crd[j][0]*d_PsiDrug_dy[j]-atom_crd[j][1]*d_PsiDrug_dx[j];
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
void Psidrug::calculate() {
  auto start_psi=high_resolution_clock::now();
  if (pbc) makeWhole();
  reset();
  int step=getStep();
  atom_crd=getPositions();
  
  #pragma omp parallel for num_threads(ngrid) reduction(+:PsiDrug)
  for (unsigned i=0; i<grids.size();i++)
  {
    //Update Grid Coordinates//
    if (step==0 and numstep==0)
    {
      if (!debug) grids[i].place_random(atom_crd,i);
      grids[i].assign_bsite_bin(atom_crd,step);
      grids[i].center_grid(atom_crd);
    }
    if (!noupdate) grids[i].update(atom_crd,step,i); //add NOUPDATE for derivatives testing

    grids[i].reset();
 
    #pragma omp parallel for num_threads(nkernelsgrid)
    for (unsigned k=0; k<nkernelsgrid; k++)
    {
      for (unsigned l=0; l<grids[i].kernels[k].kernel_points.size();l++)
      {
        int point_id=grids[i].kernels[k].kernel_points[l];
        grids[i].kernels[k].calculate_activity(grids[i].positions[point_id],atom_crd,grids[i].bsite_bin);
        grids[i].activity_vec[point_id]=grids[i].kernels[k].A;
        #pragma omp critical
        {
        grids[i].add_activity(grids[i].kernels[k].A,
                              grids[i].kernels[k].dA_dx,
                              grids[i].kernels[k].dA_dy,
                              grids[i].kernels[k].dA_dz);
        }
      }
    }
    PsiDrug+=grids[i].PsiGrid;
    for (unsigned j=0;j<n_atoms;j++)
    {
      d_PsiDrug_dx[j]+=grids[i].d_Psigrid_dx[j];
      d_PsiDrug_dy[j]+=grids[i].d_Psigrid_dy[j];
      d_PsiDrug_dz[j]+=grids[i].d_Psigrid_dz[j];
    }
    if (step==0)
    {
     grids[i].print_grid(i,step); //comment out for derivatives testing
    }
    if (step%gridstride==0)
    {
     grids[i].print_grid_stats(i,step,atoms);
     if (target) grids[i].print_grid_rmsd(i,step,atom_crd);
    }
    if(taboostride>0 and step%taboostride==0)
       grids[i].assign_bsite_visited();
  }
  setValue(PsiDrug);
  auto end_psi=high_resolution_clock::now();

  auto start_dfix=high_resolution_clock::now();
  if (!nodxfix) correct_derivatives();
  auto end_dfix=high_resolution_clock::now();

  for (unsigned j=0; j<n_atoms;j++)
  {
    setAtomsDerivatives(j,Vector(d_PsiDrug_dx[j],d_PsiDrug_dy[j],d_PsiDrug_dz[j]));
  }
  setBoxDerivativesNoPbc();
  
  if (performance)
  {
    string filename="performance.csv";
    ofstream wfile;
    if (step==0) 
    {
      wfile.open(filename.c_str());
      wfile << "step us_psi us_dxfix us_total us_psi_avg us_dxfix_avg us_total_avg elapsed" << endl;
    }
    else 
    {
    wfile.open(filename.c_str(),std::ios_base::app);
    }
    double duration_psi = duration_cast<microseconds>(end_psi-start_psi).count();
    elapsed_psi+=duration_psi;
    double avg_psi=elapsed_psi/(numstep+1);

    double duration_dfix = duration_cast<microseconds>(end_dfix-start_dfix).count();
    elapsed_dfix+=duration_dfix;
    double avg_dfix=elapsed_dfix/(numstep+1);

    double us_total = duration_psi+duration_dfix;
    double us_total_avg=avg_psi+avg_dfix;
    double elapsed = elapsed_psi+elapsed_dfix;
    double avg_total=elapsed/(numstep+1);

    wfile << numstep << " " << duration_psi << " " << duration_dfix << " " << us_total << " " << avg_psi << " " << avg_dfix << " " << avg_total << " " << elapsed << endl;
    
    wfile.close();
  }
  numstep++;
}

}
}



