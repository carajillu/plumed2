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

#define zero_tol 0.000001


Probe::Probe(double Mind_slope, double Mind_intercept, double CCMin, double CCMax,double DeltaCC, double DMin, double DeltaD, unsigned N_atoms)
{
  n_atoms=N_atoms;
  mind_slope=Mind_slope; //slope of the mind linear implementation
  mind_intercept=Mind_intercept; //intercept of the mind linear implementation
  CCmin=CCMin; // mind below which an atom is considered to be clashing with the probe 
  CCmax=CCMax; // distance above which an atom is considered to be too far away from the probe*
  deltaCC=DeltaCC; // interval over which contact terms are turned on and off
  Dmin=DMin; // packing factor below which depth term equals 0
  deltaD=DeltaD; // interval over which depth term turns from 0 to 1
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

  Son_r=vector<double>(n_atoms,0);
  dSon_r_dx=vector<double>(n_atoms,0);
  dSon_r_dy=vector<double>(n_atoms,0);
  dSon_r_dz=vector<double>(n_atoms,0);

  xyz=vector<double>(3,0);
  arma_xyz=arma::mat(1,3,arma::fill::zeros);
  centroid=vector<double>(3,0);
  centroid0=vector<double>(3,0);

  dmind_dx=vector<double>(n_atoms,0);
  dmind_dy=vector<double>(n_atoms,0);
  dmind_dz=vector<double>(n_atoms,0);

  dCC_dx=vector<double>(n_atoms,0);
  dCC_dy=vector<double>(n_atoms,0);
  dCC_dz=vector<double>(n_atoms,0);

  dD_dx=vector<double>(n_atoms,0);
  dD_dy=vector<double>(n_atoms,0);
  dD_dz=vector<double>(n_atoms,0);

  dH_dx=vector<double>(n_atoms,0);
  dH_dy=vector<double>(n_atoms,0);
  dH_dz=vector<double>(n_atoms,0);

  d_activity_dx=vector<double>(n_atoms,0);
  d_activity_dy=vector<double>(n_atoms,0);
  d_activity_dz=vector<double>(n_atoms,0);

  atomcoords_0=arma::mat(n_atoms,3,arma::fill::zeros);
  atomcoords=arma::mat(n_atoms,3,arma::fill::zeros);
  wCov=arma::mat(3,3,arma::fill::zeros);
  U=arma::mat(3,3,arma::fill::zeros);
  s=arma::vec(3,arma::fill::zeros);
  V=arma::mat(3,3,arma::fill::zeros);
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
void Probe::calculate_r(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z)
{
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
void Probe::calculate_Soff_r()
{
 total_Soff=0;
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

void Probe::calculate_Son_r()
{
 for (unsigned j=0; j<n_atoms; j++)
 {
  double m_r=COREFUNCTIONS::m_v(r[j],0,CCmax);
  double dm_dr=COREFUNCTIONS::dm_dv(CCmax);

  Son_r[j]=COREFUNCTIONS::Son_m(m_r,1);
  dSon_r_dx[j]=COREFUNCTIONS::dSon_dm(m_r,1)*dm_dr*dr_dx[j];
  dSon_r_dy[j]=COREFUNCTIONS::dSon_dm(m_r,1)*dm_dr*dr_dy[j];
  dSon_r_dz[j]=COREFUNCTIONS::dSon_dm(m_r,1)*dm_dr*dr_dz[j];
 }
}

void Probe::calculate_mind()
{
 double sum_prod=0;
 for (unsigned j=0;j<n_atoms;j++)
 {
   sum_prod+=Soff_r[j]*r[j];
 }
 if (total_Soff<zero_tol) //avoid 0/0 error
 {
   mind=0;
 }
 else
 {
   mind=mind_slope*(sum_prod/total_Soff)+mind_intercept; 
   for (unsigned j=0;j<r.size();j++)
   {
     dmind_dx[j]=mind_slope*(((dSoff_r_dx[j]*r[j]+dr_dx[j]*Soff_r[j])*total_Soff)-(dSoff_r_dx[j]*sum_prod))/pow(total_Soff,2);
     dmind_dy[j]=mind_slope*(((dSoff_r_dy[j]*r[j]+dr_dy[j]*Soff_r[j])*total_Soff)-(dSoff_r_dy[j]*sum_prod))/pow(total_Soff,2);
     dmind_dz[j]=mind_slope*(((dSoff_r_dz[j]*r[j]+dr_dz[j]*Soff_r[j])*total_Soff)-(dSoff_r_dz[j]*sum_prod))/pow(total_Soff,2);
   }
 }

 //mind_check
 /*
 double mind_real=INFINITY;
 for (unsigned j=0;j<n_atoms;j++)
 {
  if (r[j]<mind_real) mind_real=r[j];
 }
 */
}

void Probe::calculate_CC()
{
 double m=mind/CCmin;
 CC=Son_m(m,1);
 double dCC=dSon_dm(m,1)*dm_dv(CCmin);
 for (unsigned j=0;j<n_atoms;j++)
 {
   dCC_dx[j]=dCC*dmind_dx[j];
   dCC_dy[j]=dCC*dmind_dy[j];
   dCC_dz[j]=dCC*dmind_dz[j];
 }
}

void Probe::calculate_D()
{
 D=0;
 for (unsigned j=0;j<n_atoms;j++)
 {
   //cout << Son_r[j] << " " << Soff_r[j] << endl;
   D+=Son_r[j]*Soff_r[j];
   dD_dx[j]=dSon_r_dx[j]*Soff_r[j]+dSoff_r_dx[j]*Son_r[j];
   dD_dy[j]=dSon_r_dy[j]*Soff_r[j]+dSoff_r_dy[j]*Son_r[j];
   dD_dz[j]=dSon_r_dz[j]*Soff_r[j]+dSoff_r_dz[j]*Son_r[j];
 }
}

void Probe::calculate_H()
{
 double m=(D-Dmin)/deltaD;
 H=Son_m(m,1);
 double dH=dSon_dm(m,1)*dm_dv(deltaD);
 for (unsigned j=0;j<n_atoms;j++)
 {
   dH_dx[j]=dH*dD_dx[j];
   dH_dy[j]=dH*dD_dy[j];
   dH_dz[j]=dH*dD_dz[j];
 }
}

void Probe::calculate_activity(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z)
{
 calculate_r(atoms_x,atoms_y,atoms_z);
 calculate_Soff_r();
 calculate_Son_r();
 calculate_mind();
 calculate_CC();
 calculate_D();
 calculate_H();
 activity=CC*H;
 for (unsigned j=0;j<n_atoms;j++)
 {
   d_activity_dx[j]=dCC_dx[j]*H+dH_dx[j]*CC;
   d_activity_dy[j]=dCC_dy[j]*H+dH_dy[j]*CC;
   d_activity_dz[j]=dCC_dz[j]*H+dH_dz[j]*CC;
 }
}

void Probe::kabsch()
{
 //Obtain rotmat with Kabsch Algorithm
 //we want to rotate atomcoords_0 into atomcoords, and NOT the other way round
 wCov=arma::trans(atomcoords)*atomcoords_0; //calculate covariance matrix
 //SVD of wCov
 arma::svd(U,s,V,wCov);
 // Calculate R 
 //cout << "R" << endl;
 R=V*arma::trans(U);
 //R.print();
}

void Probe::move_probe(unsigned step, vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z)
{
 //calculate centroid (better with COM, maybe?)
 centroid[0]=0;
 centroid[1]=0;
 centroid[2]=0;
 for (unsigned j=0; j<n_atoms;j++)
 {
   centroid[0]+=atoms_x[j];
   centroid[1]+=atoms_y[j];
   centroid[2]+=atoms_z[j];
 }
 centroid[0]/=n_atoms;
 centroid[1]/=n_atoms;
 centroid[2]/=n_atoms;

 //cout << centroid[0] << " " << centroid[1] << " " << centroid[2] << endl;

 //remove centroid from atom coordinates
 for (unsigned j=0; j<n_atoms;j++)
  {
   if (step==0)
   {
   atomcoords_0.row(j).col(0)=atoms_x[j]-centroid[0];
   atomcoords_0.row(j).col(1)=atoms_y[j]-centroid[1];
   atomcoords_0.row(j).col(2)=atoms_z[j]-centroid[2];
   centroid0[0]=centroid[0];
   centroid0[1]=centroid[1];
   centroid0[2]=centroid[2];
   }
   atomcoords.row(j).col(0)=atoms_x[j]-centroid[0];
   atomcoords.row(j).col(1)=atoms_y[j]-centroid[1];
   atomcoords.row(j).col(2)=atoms_z[j]-centroid[2];
  }

  //Calculate rotation matrix
  kabsch();

  //cout << "probe0: " << xyz[0] << " " << xyz[1] << " " << xyz[2] << endl;
  //cout << "centroid0; " << centroid0[0] << " " << centroid0[1] << " " << centroid0[2] << endl;
  arma_xyz.row(0).col(0)=xyz[0]-centroid0[0];
  arma_xyz.row(0).col(1)=xyz[1]-centroid0[1];
  arma_xyz.row(0).col(2)=xyz[2]-centroid0[2];

  arma_xyz=arma_xyz*R;
  //arma_xyz.print();
  xyz[0]=arma::as_scalar(arma_xyz.row(0).col(0))+centroid[0];
  xyz[1]=arma::as_scalar(arma_xyz.row(0).col(1))+centroid[1];
  xyz[2]=arma::as_scalar(arma_xyz.row(0).col(2))+centroid[2];

  //cout << "probe:: " << xyz[0] << " " << xyz[1] << " " << xyz[2] << endl;
  //cout << "centroid: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << endl;

   //backup atomcoords
  for (unsigned i=0;i<3;i++)
  {
   for (unsigned j=0;j<n_atoms;j++)
   {
     atomcoords_0.row(j).col(i)=atomcoords.row(j).col(i);
   }
   centroid0[i]=centroid[i];
  }
}

void Probe::print_probe_movement(int id, int step, vector<PLMD::AtomNumber> atoms, unsigned n_atoms, double ref_x, double ref_y, double ref_z)
{
  double r=sqrt((pow((xyz[0]-ref_x),2))+(pow((xyz[1]-ref_y),2))+(pow((xyz[2]-ref_z),2)));
  string filename = "probe-";
  filename.append(to_string(id));
  //filename.append("-step-");
  //filename.append(to_string(step));
  filename.append("-stats.csv");
  ofstream wfile;
  if (step==0)
  {
   wfile.open(filename.c_str());
   //wfile << "Step j j_index Soff_r" << endl;
   wfile << "Step Dref mind CC D H Psi" << endl;
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
  wfile << step << " " << r << " " << mind << " " << CC << " " << D << " " << H << " " << activity << " " << endl;
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

