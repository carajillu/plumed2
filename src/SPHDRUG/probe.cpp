#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <iterator>
#include <armadillo> 
//you can disable bonds check at compile time with the flag -DARMA_NO_DEBUG (makes matrix multiplication 40%ish faster).

#include "core/ActionAtomistic.h"
#include "probe.h"
#include "corefunctions.h"

using namespace std;
using namespace COREFUNCTIONS;


Probe::Probe(double Rprobe, double Mind_slope, double Mind_intercept, double CCMin, double CCMax,double DeltaCC, double DMin, double DeltaD, unsigned n_atoms)
{
  rprobe=Rprobe; // radius of each spherical probe
  mind_slope=Mind_slope; //slope of the mind linear implementation
  mind_intercept=Mind_intercept; //intercept of the mind linear implementation
  CCmin=CCMin; // mind below which an atom is considered to be clashing with the probe 
  CCmax=CCMax; // distance above which an atom is considered to be too far away from the probe*
  deltaCC=DeltaCC; // interval over which contact terms are turned on and off
  Dmin=DMin; // packing factor below which depth term equals 0
  deltaD=DeltaD; // interval over which depth term turns from 0 to 1
  Pmax=Dmin+deltaD; // number of atoms surrounding the probe for it to be considered completely packed

  //allocate vectors
  rx=vector<double>(n_atoms,0);
  ry=vector<double>(n_atoms,0);
  rz=vector<double>(n_atoms,0);

  r=vector<double>(n_atoms,0);
  dr_dx=vector<double>(n_atoms,0);
  dr_dy=vector<double>(n_atoms,0);
  dr_dz=vector<double>(n_atoms,0);

  Soff_r=vector<double>(n_atoms,0);
  dSoff_r_dx=vector<double>(n_atoms,0);
  dSoff_r_dy=vector<double>(n_atoms,0);
  dSoff_r_dz=vector<double>(n_atoms,0);

  xyz=vector<double>(3,0);
  centroid=vector<double>(3,0);

  atomcoords_0=arma::mat(n_atoms,3,arma::fill::zeros);
  atomcoords=arma::mat(n_atoms,3,arma::fill::zeros); //This is directly transposed
  weights=arma::mat(n_atoms,n_atoms,arma::fill::zeros);
  wCov=arma::mat(3,3,arma::fill::zeros);
  eigenvalues=arma::vec(3,arma::fill::zeros);
  eigenvectors=arma::mat(3,3,arma::fill::zeros);
  eigenvectors_sorted=arma::mat(3,3,arma::fill::zeros);
  B=arma::mat(3,3,arma::fill::zeros);
  R=arma::mat(3,3,arma::fill::zeros); //Rotation matrix. filled with zeros to make sure step 0 gives Identity Matrix
}

// Place probe on top a specified or randomly chosen atom, only at step 0
void Probe::place_probe(double x, double y, double z)
{
  xyz[0]=x;
  xyz[1]=y;
  xyz[2]=z;
}

//calculate distance between the center of the probe and the atoms, and all their derivatives
void Probe::calculate_r(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z, unsigned n_atoms)
{
 #pragma omp parallel for
 for (unsigned j=0; j<n_atoms; j++)
 {
   rx[j]=atoms_x[j]-xyz[0];
   ry[j]=atoms_y[j]-xyz[1];
   rz[j]=atoms_z[j]-xyz[2];

   r[j]=sqrt(pow(rx[j],2)+pow(ry[j],2)+pow(rz[j],2));

   dr_dx[j]=rx[j]/r[j];
   dr_dy[j]=ry[j]/r[j];
   dr_dz[j]=rz[j]/r[j];
 }
}

//Calculate Soff_r and all their derivatives
void Probe::calculate_Soff_r(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z, unsigned n_atoms)
{
 total_Soff=0;
 #pragma omp parallel for reduction(+:total_Soff)
 for (unsigned j=0; j<n_atoms; j++)
 {
  double m_r=COREFUNCTIONS::m_v(r[j],CCmax,deltaCC);
  double dm_dr=COREFUNCTIONS::dm_dv(deltaCC);

  Soff_r[j]=COREFUNCTIONS::Soff_m(m_r,1);
  total_Soff+=Soff_r[j];
  dSoff_r_dx[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dx[j];
  dSoff_r_dy[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dy[j];
  dSoff_r_dz[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dz[j];
 }
}

//calculate a weighted centroid as a reference to move the probe
void Probe::calc_centroid(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z, unsigned n_atoms)
{
 centroid[0]=0;
 centroid[1]=0;
 centroid[2]=0;
 
 for (unsigned j=0; j<n_atoms;j++)
 {
   centroid[0]+=atoms_x[j]*Soff_r[j];
   centroid[1]+=atoms_y[j]*Soff_r[j];
   centroid[2]+=atoms_z[j]*Soff_r[j];
   total_Soff+=Soff_r[j];
 }
 centroid[0]/=total_Soff;
 centroid[1]/=total_Soff;
 centroid[2]/=total_Soff;
 /*
for (unsigned j=0; j<n_atoms;j++)
 {
   centroid[0]+=atoms_x[j];
   centroid[1]+=atoms_y[j];
   centroid[2]+=atoms_z[j];
   total_Soff+=1;
 }
 centroid[0]/=total_Soff;
 centroid[1]/=total_Soff;
 centroid[2]/=total_Soff;
 */
}

void Probe::kabsch(unsigned step, vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z, unsigned n_atoms, vector<double> masses, double total_mass)
{
 for (unsigned j=0; j<n_atoms;j++)
 {
   if (step==0)
   {
   atomcoords_0.row(j).col(0)=atoms_x[j]-centroid[0];
   atomcoords_0.row(j).col(1)=atoms_y[j]-centroid[1];
   atomcoords_0.row(j).col(2)=atoms_z[j]-centroid[2];
   }
   atomcoords.row(j).col(0)=atoms_x[j]-centroid[0];
   atomcoords.row(j).col(1)=atoms_y[j]-centroid[1];
   atomcoords.row(j).col(2)=atoms_z[j]-centroid[2];

   weights.row(j).col(j)=Soff_r[j]; //This can be memory expensive!
 }

 //Obtain rotmat with Kabsch Algorithm
 //we want to rotate atomcoords_0 into atomcoords, and NOT the other way round

 //wCov=arma::trans(atomcoords)*atomcoords_0; //calculate weighted covariance matrix
 wCov=arma::trans(atomcoords)*weights*atomcoords_0; //calculate weighted covariance matrix
 cout << "wCov" << endl;
 wCov.print();
 cout << "wCov2" << endl;
 wCov=wCov*arma::trans(wCov); //square the weighted covariance matrix (it is symmetrical so its transposed is the same)
 wCov.print();

 
 //SVD of wCov
 arma::mat U;
 arma::vec s;
 arma::mat V;

 arma::svd(U,s,V,wCov);
 U.print();
 s.print();
 V.print();

 // Calculate R (finally!)
 cout << "R" << endl;
 R=V*arma::trans(U);
 R.print();
 
  
 //backup atomcoords
 atomcoords_0=atomcoords;
 
 
}

void Probe::print_probe_movement(int id, int step, vector<PLMD::AtomNumber> atoms, unsigned n_atoms, double ref_x, double ref_y, double ref_z)
{
  double r=sqrt((pow((xyz[0]-ref_x),2))+(pow((xyz[1]-ref_y),2))+(pow((xyz[2]-ref_z),2)));
  string filename = "probe-";
  filename.append(to_string(id));
  //filename.append("-step-");
  //filename.append(to_string(step));
  filename.append("-movement.csv");
  ofstream wfile;
  if (step==0)
  {
   wfile.open(filename.c_str());
   //wfile << "Step j j_index Soff_r" << endl;
   wfile << "Dref" << endl;
  }
  else
  {
   wfile.open(filename.c_str(),std::ios_base::app);
  }
  /*
  for (unsigned j=0; j<n_atoms; j++)
  {
   if (Soff_r[j]>0.000001) //many doubles are gonna be different than 0
       wfile << step << " " << j << " " << atoms[j].index() << " " << Soff_r[j] << endl;
  }
  */
  wfile << r << endl;
  wfile.close();
}

void Probe::print_probe_xyz(int id, int step)
{
 string filename = "probe-";
 filename.append(to_string(id));
 filename.append(".xyz");
 ofstream wfile;
 if (step==0)
 {
  wfile.open(filename.c_str());
 }
 else
 {
  wfile.open(filename.c_str(),std::ios_base::app);
 }
 wfile << 1 << endl;
 wfile << "Probe  "<< to_string(id) << endl;
 wfile << "Ge " << std::fixed << std::setprecision(5) << xyz[0]*10 << " " << xyz[1]*10 << " " << xyz[2]*10 << endl;
 wfile.close();
}

