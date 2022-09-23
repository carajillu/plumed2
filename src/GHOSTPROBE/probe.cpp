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


Probe::Probe(unsigned Probe_id, 
            double RMin, double DeltaRmin, 
            double RMax, double DeltaRmax, 
            double phimin, double deltaphi, 
            double psimin, double deltapsi, 
            unsigned N_atoms, double kpert, 
            unsigned Init_j)
{
  init_j=Init_j;
  probe_id=Probe_id;
  dxcalc=true;
  n_atoms=N_atoms;
  Rmin=RMin; // mind below which an atom is considered to be clashing with the probe 
  deltaRmin=DeltaRmin; // interval over which contact terms are turned on and off
  Rmax=RMax; // distance above which an atom is considered to be too far away from the probe*
  deltaRmax=DeltaRmax; // interval over which contact terms are turned on and off
  Cmin=phimin; 
  deltaC=deltaphi;
  Pmin=psimin; 
  deltaP=deltapsi;
  Kpert=kpert;
  //allocate vectors
  rx=vector<double>(n_atoms,0);
  ry=vector<double>(n_atoms,0);
  rz=vector<double>(n_atoms,0);

  r=vector<double>(n_atoms,0);
  dr_dx=vector<double>(n_atoms,0);
  dr_dy=vector<double>(n_atoms,0);
  dr_dz=vector<double>(n_atoms,0);

  enclosure=vector<double>(n_atoms,0);
  total_enclosure=0;
  d_enclosure_dx=vector<double>(n_atoms,0);
  d_enclosure_dy=vector<double>(n_atoms,0);
  d_enclosure_dz=vector<double>(n_atoms,0);

  clash=vector<double>(n_atoms,0);
  total_clash=0;
  d_clash_dx=vector<double>(n_atoms,0);
  d_clash_dy=vector<double>(n_atoms,0);
  d_clash_dz=vector<double>(n_atoms,0);

  C=0;
  dC_dx=vector<double>(n_atoms,0);
  dC_dy=vector<double>(n_atoms,0);
  dC_dz=vector<double>(n_atoms,0);

  P=0;
  dP_dx=vector<double>(n_atoms,0);
  dP_dy=vector<double>(n_atoms,0);
  dP_dz=vector<double>(n_atoms,0);

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
  if (((activity_cum > activity_old) and step>0))
  {
   activity_old=activity_cum;
   activity_cum=0;
   ptries=0;
   return;
  }

  dxcalc=false; // switch off derivatives calculation during the perturbation trials
  xyz0=xyz;
  ptries=0;
  total_enclosure=0;
  double r=INFINITY;

  while (total_enclosure<Pmin)
  {
    if (ptries == 10000) 
    {
      cout << "Step " << step << ": probe " << probe_id << " could not be settled after " << ptries << " perturbation trials." << endl;
      cout << "enclosure    clash    activity    Ptries" << endl;
      cout << total_enclosure << "    " << total_clash << "    " << activity << "    " << ptries << endl;
      //cout << "Simulation will now terminate" << endl;
      //exit(0);
      xyz[0]=atoms_x[j_min_r];
      xyz[0]=atoms_x[j_min_r];
      xyz[0]=atoms_x[j_min_r];
      calc_pert();
      break;
    }
    xyz=xyz0;
    calc_pert();
    calculate_r(atoms_x,atoms_y,atoms_z);
    calculate_enclosure();
    r=sqrt((pow((xyz[0]-atoms_x[init_j]),2))+(pow((xyz[1]-atoms_y[init_j]),2))+(pow((xyz[2]-atoms_z[init_j]),2)));
    ptries++;
  }
  activity_old=activity_cum;
  activity_cum=0;
  dxcalc=true; //switch derivatives calculation back on
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
   if (dxcalc)
   {
   dr_dx[j]=rx[j]/r[j];
   dr_dy[j]=ry[j]/r[j];
   dr_dz[j]=rz[j]/r[j];
   }

   if (r[j]<min_r)
   {
    min_r=r[j];
    j_min_r=j;
   }
 }
}

//Calculate Soff_r and all their derivatives
void Probe::calculate_enclosure()
{
 total_enclosure=0;
 for (unsigned j=0; j<n_atoms; j++)
 {
  if (r[j] >= (Rmax+deltaRmax))
  {
   enclosure[j]=0;
   d_enclosure_dx[j]=0;
   d_enclosure_dy[j]=0;
   d_enclosure_dz[j]=0;
   continue;
  }

  double m_r=COREFUNCTIONS::m_v(r[j],Rmax,deltaRmax);
  double dm_dr=COREFUNCTIONS::dm_dv(deltaRmax);

  enclosure[j]=COREFUNCTIONS::Soff_m(m_r,1);
  if (dxcalc)
  {
  d_enclosure_dx[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dx[j];
  d_enclosure_dy[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dy[j];
  d_enclosure_dz[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dz[j];
  }
  total_enclosure+=enclosure[j];
 }
}

void Probe::calculate_P()
{
 calculate_enclosure();
 double m_enclosure=COREFUNCTIONS::m_v(total_enclosure,Pmin,deltaP);
 double dm_enclosure=COREFUNCTIONS::dm_dv(deltaP);
 P=COREFUNCTIONS::Son_m(m_enclosure,1);
 if (dxcalc)
 {
  for (unsigned j=0; j<n_atoms; j++)
  {
   dP_dx[j]=dSon_dm(m_enclosure,1)*dm_enclosure*d_enclosure_dx[j];
   dP_dy[j]=dSon_dm(m_enclosure,1)*dm_enclosure*d_enclosure_dy[j];
   dP_dz[j]=dSon_dm(m_enclosure,1)*dm_enclosure*d_enclosure_dz[j];
  }
 }
}

void Probe::calculate_clash()
{
 total_clash=0; 
 for (unsigned j=0; j<n_atoms; j++)
 {
  if (r[j] >= Rmin+deltaRmin)
  {
   clash[j]=0;
   d_clash_dx[j]=0;
   d_clash_dy[j]=0;
   d_clash_dz[j]=0;
   continue;
  }
  double m_r=COREFUNCTIONS::m_v(r[j],Rmin,deltaRmin);
  double dm_dr=COREFUNCTIONS::dm_dv(deltaRmin);

  clash[j]=COREFUNCTIONS::Soff_m(m_r,1);
  if (dxcalc)
  {
  d_clash_dx[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dx[j];
  d_clash_dy[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dy[j];
  d_clash_dz[j]=COREFUNCTIONS::dSoff_dm(m_r,1)*dm_dr*dr_dz[j];
  }
  total_clash+=clash[j];
 }
}

void Probe::calculate_C()
{
 calculate_clash();
 double m_clash=COREFUNCTIONS::m_v(total_clash,Cmin,deltaC);
 double dm_clash=COREFUNCTIONS::dm_dv(deltaC);
 C=COREFUNCTIONS::Soff_m(m_clash,1);
 if (dxcalc)
 {
  for (unsigned j=0; j<n_atoms; j++)
   {
    dC_dx[j]=COREFUNCTIONS::dSoff_dm(m_clash,1)*dm_clash*d_clash_dx[j];
    dC_dy[j]=COREFUNCTIONS::dSoff_dm(m_clash,1)*dm_clash*d_clash_dy[j];
    dC_dz[j]=COREFUNCTIONS::dSoff_dm(m_clash,1)*dm_clash*d_clash_dz[j];
   }
 }
}

void Probe::calculate_activity(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z)
{
 calculate_r(atoms_x,atoms_y,atoms_z);
 calculate_C();
 calculate_P();
 activity=C*P;
 if (dxcalc)
 {
  for (unsigned j=0; j<n_atoms;j++)
  {
   d_activity_dx[j]=C*dP_dx[j]+P*dC_dx[j];
   d_activity_dy[j]=C*dP_dy[j]+P*dC_dy[j];
   d_activity_dz[j]=C*dP_dz[j]+P*dC_dz[j];
  }
 }
 activity_cum+=activity;
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

void Probe::print_probe_movement(int id, int step, vector<PLMD::AtomNumber> atoms, unsigned n_atoms)
{
  string filename = "probe-";
  filename.append(to_string(id));
  //filename.append("-step-");
  //filename.append(to_string(step));
  filename.append("-stats.csv");
  ofstream wfile;
  wfile.open(filename.c_str(),std::ios_base::app);
  if (step==0)
  {
   wfile << "Step min_r_serial min_r enclosure P clash C activity activity_cum activity_old Ptries" << endl;
  }
  /*
  for (unsigned j=0; j<n_atoms; j++)
  {
   if (Soff_r[j]>0.000001) //many doubles are gonna be different than 0
       wfile << step << " " << j << " " << atoms[j].index() << " " << Soff_r[j] << endl;
  }
  */
  wfile << step << " " << atoms[j_min_r].serial() << " " << min_r << " " 
        << total_enclosure << " " << P << " " 
        << total_clash << " " << C << " " 
        << activity << " " << activity_cum << " " << activity_old << " "
        << ptries << endl;
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

