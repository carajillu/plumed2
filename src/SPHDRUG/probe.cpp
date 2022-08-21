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


Probe::Probe(double CCMin, double CCMax, double DeltaCC, double phimin, double deltaphi, unsigned N_atoms, double kpert)
{
  n_atoms=N_atoms;
  CCmin=CCMin; // mind below which an atom is considered to be clashing with the probe 
  CCmax=CCMax; // distance above which an atom is considered to be too far away from the probe*
  deltaCC=DeltaCC; // interval over which contact terms are turned on and off
  Phimin=phimin; // packing factor below which depth term equals 0
  deltaPhi=deltaphi; // interval over which depth term turns from 0 to 1
  Kpert=kpert;
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

  Phi=0;
  dPhi_dx=vector<double>(n_atoms,0);
  dPhi_dy=vector<double>(n_atoms,0);
  dPhi_dz=vector<double>(n_atoms,0);

  xyz=vector<double>(3,0);
  xyz_pert=vector<double>(3,0);
  xyz0=vector<double>(3,0);
  ptries=0;
  arma_xyz=arma::mat(1,3,arma::fill::zeros);
  centroid=vector<double>(3,0);
  centroid0=vector<double>(3,0);

  d_activity_dx=vector<double>(n_atoms,0);
  d_activity_dy=vector<double>(n_atoms,0);
  d_activity_dz=vector<double>(n_atoms,0);

  atomcoords_0=arma::mat(n_atoms,3,arma::fill::zeros);
  atomcoords=arma::mat(n_atoms,3,arma::fill::zeros);
  wCov=arma::mat(3,3,arma::fill::zeros);
  weights=arma::mat(n_atoms,n_atoms,arma::fill::eye);
  U=arma::mat(3,3,arma::fill::zeros);
  s=arma::vec(3,arma::fill::zeros);
  V=arma::mat(3,3,arma::fill::zeros);
  R=arma::mat(3,3,arma::fill::zeros); //Rotation matrix. filled with zeros to make sure step 0 gives Identity Matrix
}

// Place probe on top a specified or randomly chosen atom, only at step 0. Offsetting the probe so it doesn't fall right on top of the atom (seems more stable)
void Probe::place_probe(double x, double y, double z)
{
  xyz[0]=x;
  xyz[1]=y;
  xyz[2]=z;
}

void Probe::calc_pert()
{
  random_device rd;  // only used once to initialise (seed) engine
  mt19937 rng(rd()); // random-number engine used (Mersenne-Twister in this case)

  for (unsigned i=0; i<3;i++)
  {
   uniform_real_distribution<double> uni(-1, 1); // guaranteed unbiased
   auto random_double = uni(rng);
   xyz_pert[i]=random_double;
  }
  double norm=sqrt(pow(xyz_pert[0],2)+pow(xyz_pert[1],2)+pow(xyz_pert[2],2));
  double k=Kpert/norm;

  xyz_pert[0]*=k;
  xyz_pert[1]*=k;
  xyz_pert[2]*=k;
  xyz[0]+=xyz_pert[0];
  xyz[1]+=xyz_pert[1];
  xyz[2]+=xyz_pert[2];
}

void Probe::perturb_probe(unsigned step, vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z)
{
  //if no perturbation is needed, just update data and leave
  if ((activity_cum >= activity_old) and step>0)
  {
   activity_old=activity_cum;
   activity_cum=0;
   ptries=0;
   return;
  }
  xyz0=xyz;
  ptries=0;
  activity=0;
  while (activity<0.00001)
  {
    xyz=xyz0;
    calc_pert();
    calculate_activity(atoms_x,atoms_y,atoms_z);
    ptries++;
    if (ptries > 10000) 
    {
      cout << "Probe could not be settled after 10000 perturbation trials. Exiting." << endl;
      exit(0);
    }
    activity_old=activity_cum;
    activity_cum=0;
  }    
}

//calculate distance between the center of the probe and the atoms, and all their derivatives
void Probe::calculate_r(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z)
{
 min_r=INFINITY;
 j_min_r=INFINITY; 
 for (unsigned j=0; j<n_atoms; j++)
 {
   rx[j]=atoms_x[j]-xyz[0];
   ry[j]=atoms_y[j]-xyz[1];
   rz[j]=atoms_z[j]-xyz[2];

   r[j]=sqrt(pow(rx[j],2)+pow(ry[j],2)+pow(rz[j],2));

   dr_dx[j]=rx[j]/r[j];
   dr_dy[j]=ry[j]/r[j];
   dr_dz[j]=rz[j]/r[j];

   if (r[j]<min_r)
   {
    min_r=r[j];
    j_min_r=j;
   }
 }
}

//Calculate Soff_r and all their derivatives
void Probe::calculate_Soff_r()
{
 for (unsigned j=0; j<n_atoms; j++)
 {
  if (r[j] >= (CCmax+deltaCC))
  {
   Soff_r[j]=0;
   dSoff_r_dx[j]=0;
   dSoff_r_dy[j]=0;
   dSoff_r_dz[j]=0;
   continue;
  }
  double m_r=COREFUNCTIONS::m_v(r[j],CCmax,deltaCC);
  double dm_dr=COREFUNCTIONS::dm_dv(deltaCC);

  Soff_r[j]=COREFUNCTIONS::Soff_m(m_r,1);
  dSoff_r_dx[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dx[j];
  dSoff_r_dy[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dy[j];
  dSoff_r_dz[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dz[j];
 }
}

void Probe::calculate_Son_r()
{
 for (unsigned j=0; j<n_atoms; j++)
 {
  if (r[j] <= CCmin)
  {
   Son_r[j]=0;
   dSon_r_dx[j]=0;
   dSon_r_dy[j]=0;
   dSon_r_dz[j]=0;
   continue;
  }
  double m_r=COREFUNCTIONS::m_v(r[j],CCmin,deltaCC);
  double dm_dr=COREFUNCTIONS::dm_dv(CCmin);

  Son_r[j]=COREFUNCTIONS::Son_m(m_r,1);
  dSon_r_dx[j]=COREFUNCTIONS::dSon_dm(m_r,1)*dm_dr*dr_dx[j];
  dSon_r_dy[j]=COREFUNCTIONS::dSon_dm(m_r,1)*dm_dr*dr_dy[j];
  dSon_r_dz[j]=COREFUNCTIONS::dSon_dm(m_r,1)*dm_dr*dr_dz[j];
 }
}

void Probe::calculate_Phi()
{
 Phi=0;
 for (unsigned j=0; j<n_atoms; j++)
 {
  Phi+=Son_r[j]*Soff_r[j];
  dPhi_dx[j]=Son_r[j]*dSoff_r_dx[j]+Soff_r[j]*dSon_r_dx[j];
  dPhi_dy[j]=Son_r[j]*dSoff_r_dy[j]+Soff_r[j]*dSon_r_dy[j];
  dPhi_dz[j]=Son_r[j]*dSoff_r_dz[j]+Soff_r[j]*dSon_r_dz[j];
 }
}

void Probe::calculate_activity(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z)
{
 calculate_r(atoms_x,atoms_y,atoms_z);
 calculate_Soff_r();
 calculate_Son_r();
 calculate_Phi();
 double m_Phi=COREFUNCTIONS::m_v(Phi,Phimin,deltaPhi);
 double dm_dPhi=COREFUNCTIONS::dm_dv(deltaPhi);
 activity=COREFUNCTIONS::Son_m(m_Phi,1);
 for (unsigned j=0; j<n_atoms;j++)
 {
  d_activity_dx[j]=COREFUNCTIONS::dSon_dm(m_Phi,1)*dm_dPhi*dPhi_dx[j];
  d_activity_dy[j]=COREFUNCTIONS::dSon_dm(m_Phi,1)*dm_dPhi*dPhi_dy[j];
  d_activity_dz[j]=COREFUNCTIONS::dSon_dm(m_Phi,1)*dm_dPhi*dPhi_dz[j];
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
  wfile.open(filename.c_str(),std::ios_base::app);
  if (step==0)
  {
   wfile << "Step Dref min_r Phi Psi Ptries" << endl;
  }
  /*
  for (unsigned j=0; j<n_atoms; j++)
  {
   if (Soff_r[j]>0.000001) //many doubles are gonna be different than 0
       wfile << step << " " << j << " " << atoms[j].index() << " " << Soff_r[j] << endl;
  }
  */
  wfile << step << " " << r << " " << min_r << " " << Phi << " " << activity << " " << ptries << endl;
  wfile.close();
}

void Probe::print_probe_xyz(int id, int step)
{
 string filename = "probe-";
 filename.append(to_string(id));
 filename.append(".xyz");
 ofstream wfile;
 wfile.open(filename.c_str(),std::ios_base::app);
 wfile << 1 << endl;
 wfile << "Probe  "<< to_string(id) << endl;
 wfile << "Ge " << std::fixed << std::setprecision(5) << xyz[0]*10 << " " << xyz[1]*10 << " " << xyz[2]*10 << endl;
 wfile.close();
}

