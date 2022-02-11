#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <iterator>

#include "core/ActionAtomistic.h"
#include "probe.h"
#include "corefunctions.h"

using namespace std;
using namespace COREFUNCTIONS;

Probe::Probe(double Rprobe, double Mind_slope, double Mind_intercept, double CCMin, double CCMax,double DeltaCC, double DMin, double DeltaD)
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
}

// Place probe on top a specified or randomly chosen atom, only at step 0
void Probe::place_probe(double x, double y, double z)
{
  xyz[0]=x;
  xyz[1]=y;
  xyz[2]=z;
}

//calculate a weighted centroid as a reference to move the probe
void Probe::calc_centroid(double* atoms_x, double* atoms_y, double* atoms_z, unsigned n_atoms)
{
 double x=0;
 double y=0;
 double z=0;
 double total_bsite=0;

 #pragma omp parallel for reduction(+:x,y,z,total_bsite)
 for (unsigned j=0; j<n_atoms; j++)
 {
   x+=atoms_x[j]*Soff_r[j];
   y+=atoms_y[j]*Soff_r[j];
   z+=atoms_z[j]*Soff_r[j];
   total_bsite+=Soff_r[j];
 }
 
 centroid[0]=x/total_bsite;
 centroid[1]=y/total_bsite;
 centroid[2]=z/total_bsite;
}

//calculate distance between the center of the probe and the atoms, and all their derivatives
void Probe::calculate_r(double* atoms_x, double* atoms_y, double* atoms_z, unsigned n_atoms)
{
 #pragma omp parallel for
 for (unsigned j=0; j<n_atoms; j++)
 {
   rx[j]=atoms_x[j]-xyz[0];
   ry[j]=atoms_y[j]-xyz[1];
   rz[j]=atoms_z[j]-xyz[2];

   r[j]=sqrt(pow(rx[j],2)+pow(rz[j],2)+pow(rz[j],2));

   dr_dx[j]=rx[j]/r[j];
   dr_dy[j]=ry[j]/r[j];
   dr_dz[j]=rz[j]/r[j];
 }
}

//Calculate Soff_r and all their derivatives
void Probe::calculate_Soff_r(double* atoms_x, double* atoms_y, double* atoms_z, unsigned n_atoms)
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

void Probe::move_probe(unsigned step, double* atoms_x, double* atoms_y, double* atoms_z, unsigned n_atoms, double* masses, double total_mass)
{
cout << "******************************* step " << step << " ****************************" << endl;
double delta_x=0;
double delta_y=0;
double delta_z=0;
double total_Soff=0;
for (unsigned j=0; j<n_atoms; j++)
{
  if (step==0)
  {
    atoms_x0[j]=atoms_x[j];
    atoms_y0[j]=atoms_y[j];
    atoms_z0[j]=atoms_z[j];
    xyz0[0]=xyz[0];
    xyz0[1]=xyz[1];
    xyz0[2]=xyz[2];
  }

  delta_x+=(atoms_x[j]-atoms_x0[j])*Soff_r[j];
  delta_y+=(atoms_y[j]-atoms_y0[j])*Soff_r[j];
  delta_z+=(atoms_z[j]-atoms_z0[j])*Soff_r[j];
  total_Soff+=Soff_r[j];
}

delta_x/=total_Soff;
delta_y/=total_Soff;
delta_z/=total_Soff;

xyz[0]+=delta_x;
xyz[1]+=delta_y;
xyz[2]+=delta_z;
cout << xyz[0] << " " << xyz[1] << " " << xyz[2] << endl;

for (unsigned j=0; j< n_atoms; j++)
{
atoms_x0[j]=atoms_x[j];
atoms_y0[j]=atoms_y[j];
atoms_z0[j]=atoms_z[j];
}

return;
// https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

//reset
com[0]=0;
com[1]=0;
com[2]=0;
centroid[0]=0;
centroid[1]=0;
centroid[2]=0;
total_Soff=0;

cout << "------------------------Soff-----------------------------" << endl;

//Calculate com and centroid
for (unsigned j=0;j<n_atoms;j++)
{
 com[0]+=atoms_x[j]*masses[j];
 com[1]+=atoms_y[j]*masses[j];
 com[2]+=atoms_z[j]*masses[j];
 centroid[0]+=atoms_x[j]*Soff_r[j];
 centroid[1]+=atoms_y[j]*Soff_r[j];
 centroid[2]+=atoms_z[j]*Soff_r[j];
 total_Soff+=Soff_r[j];
 if (Soff_r[j]>0.00000001)
    cout << j << " " << Soff_r[j] << endl;
}
cout << "Total " << total_Soff << endl;
cout << "------------------------Soff-----------------------------" << endl;
com[0]/=total_mass;
com[1]/=total_mass;
com[2]/=total_mass;

centroid[0]/=total_Soff;
centroid[1]/=total_Soff;
centroid[2]/=total_Soff;

//back up original com and centroid
for (unsigned i=0; i<3;i++)
{
  com_bckp[i]=com[i];
  centroid_bckp[i]=centroid[i];
}

if (step==0)
{
 xyz0[0]=xyz[0];
 xyz0[1]=xyz[1];
 xyz0[2]=xyz[2];
 com0[0]=com[0];
 com0[1]=com[1];
 com0[2]=com[2];
 centroid0[0]=centroid[0];
 centroid0[1]=centroid[1];
 centroid0[2]=centroid[2];
 comcen0[0]=comcen[0];
 comcen0[1]=comcen[1];
 comcen0[2]=comcen[2];
 //return;
}

cout << "centroid " << centroid[0] << " " << centroid[1] << " " << centroid[2] << " " << endl;
cout << "centroid0 " << centroid0[0] << " " << centroid0[1] << " " << centroid0[2] << " " << endl;

cout << "com " << com[0] << " " << com[1] << " " << com[2] << " " << endl;
cout << "com0 " << com0[0] << " " << com0[1] << " " << com0[2] << " " << endl;

//move com0 centroid0 by deltacom and xyz0 by deltacentroid

double deltacom[3];
deltacom[0]=com[0]-com0[0];
deltacom[1]=com[1]-com0[1];
deltacom[2]=com[2]-com0[2];


cout << "deltacom: " << deltacom[0] << " " << deltacom[1] << " " << deltacom[2] << endl;

for (unsigned i=0; i<3; i++)
{
  com0[i]+=deltacom[i];
  centroid0[i]+=deltacom[i];
  xyz0[i]+=deltacom[i];
}

//Remove com from com, centroid, com0 and centroid0
for (unsigned i=0; i<3; i++)
{
  com0[i]-=com[i];
  centroid[i]-=com[i];
  centroid0[i]-=com[i];
  com[i]-=com[i]; // sanity check, we can just set it to 0 if it works. THAT GOES ALWAYS LAST!!!
}

cout << "com " << com[0] << " " << com[1] << " " << com[2] << " " << endl;
cout << "com0 " << com0[0] << " " << com0[1] << " " << com0[2] << " " << endl;

// calculate vector from com to centroid and normalise to unit length
comcen_norm=0;
comcen0_norm=0;
for (unsigned i=0; i<3; i++)
{
  comcen[i]=centroid[i]-com[i];
  comcen_norm+=pow(comcen[i],2);
  comcen0[i]=centroid0[i]-com0[i];
  comcen0_norm+=pow(comcen0[i],2);
}
comcen_norm=sqrt(comcen_norm);
comcen0_norm=sqrt(comcen0_norm);

for (unsigned i=0; i<3; i++)
{
  comcen[i]/=comcen_norm;
  comcen0[i]/=comcen0_norm;
}
cout << "comcen " << comcen[0] << " " << comcen[1] << " " << comcen[2] << " " << endl;
cout << "comcen0 " << comcen0[0] << " " << comcen0[1] << " " << comcen0[2] << " " << endl;

//Calculate rotation matrix (is this R the same for normalised an unnormalised vectors?)

if (abs(comcen0[0]+comcen[0])<0.0000001 and abs(comcen0[1]+comcen[1])<0.0000001 and abs(comcen0[2]+comcen[2])<0.0000001)
 {
   
   cout << "Vectors comcen and comcen0 have opposite directions. Update probe differently not implemented." << endl;
   exit(0);
 }

//Calculate dot and cross product (direction?)
cross[0]=(comcen0[1]*comcen[2]-comcen0[2]*comcen[1]);
cross[1]=(comcen0[2]*comcen[0]-comcen0[0]*comcen[2]);
cross[2]=(comcen0[0]*comcen[1]-comcen0[1]*comcen[0]);

double dot=(comcen[0]*comcen0[0]+comcen[1]*comcen0[1]+comcen[2]*comcen0[2]);

cout << "cross " << cross[0] << " " << cross[1] << " " << cross[2] << " " << endl;
cout << "dot = " << dot << endl;

//calculate rotation matrix
double k=1/(1+dot);


// if A^2==AA

rotmat[0][0] = 1 + 0        - k*(pow(cross[1],2)+pow(cross[2],2));
rotmat[0][1] = 0 - cross[2] + k*(cross[0]*cross[1]); 
rotmat[0][2] = 0 + cross[1] + k*(cross[0]*cross[2]);
rotmat[1][0] = 0 + cross[2] + k*(cross[0]*cross[1]);
rotmat[1][1] = 1 + 0        - k*(pow(cross[0],2)+pow(cross[2],2));
rotmat[1][2] = 0 - cross[0] + k*(cross[1]*cross[2]);
rotmat[2][0] = 0 - cross[1] + k*(cross[0]*cross[2]);
rotmat[2][1] = 0 + cross[0] + k*(cross[1]*cross[2]);
rotmat[2][2] = 1 + 0        - k*(pow(cross[0],2)+pow(cross[1],2));


cout << rotmat[0][0] << " " << rotmat[0][1] << " " << rotmat[0][2] << endl;
cout << rotmat[1][0] << " " << rotmat[1][1] << " " << rotmat[1][2] << endl;
cout << rotmat[2][0] << " " << rotmat[2][1] << " " << rotmat[2][2] << endl;

//sanity check
double sanity[3];

sanity[0]=comcen0[0]*rotmat[0][0]+comcen0[1]*rotmat[0][1]+comcen0[2]*rotmat[0][2];
sanity[1]=comcen0[0]*rotmat[1][0]+comcen0[1]*rotmat[1][1]+comcen0[2]*rotmat[1][2];
sanity[2]=comcen0[0]*rotmat[2][0]+comcen0[1]*rotmat[2][1]+comcen0[2]*rotmat[2][2];

cout << " sanity = " << sanity[0] << " " << sanity[1] << " " << sanity[2] << endl;

/*
All the way down to here seems to work OK - sanity check is passed (sanity and comcen should be the same)
From this line down it's still broken
*/

// Apply rotmat to xyz (we already moved it by deltacom)
xyz[0]=xyz0[0]*rotmat[0][0]+xyz0[1]*rotmat[0][1]+xyz0[2]*rotmat[0][2];
xyz[1]=xyz0[0]*rotmat[1][0]+xyz0[1]*rotmat[1][1]+xyz0[2]*rotmat[1][2];
xyz[2]=xyz0[0]*rotmat[2][0]+xyz0[1]*rotmat[2][1]+xyz0[2]*rotmat[2][2];


cout << " xyz0 = " << xyz0[0] << " " << xyz0[1] << " " << xyz0[2] << endl;
cout << " xyz = " << xyz[0] << " " << xyz[1] << " " << xyz[2] << endl;

// backup probe, centroid and comcen coordinates
for (unsigned i=0; i<3; i++)
{
  xyz0[i]=xyz[i];
  centroid0[i]=centroid_bckp[i];
  com0[i]=com_bckp[i];
}

}

void Probe::print_probe_movement(int id, int step, vector<PLMD::AtomNumber> atoms, unsigned n_atoms)
{
  string filename = "probe-";
  filename.append(to_string(id));
  //filename.append("-step-");
  //filename.append(to_string(step));
  filename.append("-movement.csv");
  ofstream wfile;
  if (step==0)
  {
   wfile.open(filename.c_str());
   wfile << "Step j j_index Soff_r" << endl;
  }
  else
  {
   wfile.open(filename.c_str(),std::ios_base::app);
  }
  for (unsigned j=0; j<n_atoms; j++)
  {
   if (Soff_r[j]>0.000001) //many doubles are gonna be different than 0
       wfile << step << " " << j << " " << atoms[j].index() << " " << Soff_r[j] << endl;
  }
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
 wfile << 3 << endl;
 wfile << "Probe  "<< to_string(id) << endl;
 wfile << "Ge " << std::fixed << std::setprecision(5) << xyz[0]*10 << " " << xyz[1]*10 << " " << xyz[2]*10 << endl;
 wfile << "O " << std::fixed << std::setprecision(5) << centroid0[0]*10 << " " << centroid0[1]*10 << " " << centroid0[2]*10 << endl;
 wfile << "N " << std::fixed << std::setprecision(5) << com0[0]*10 << " " << com0[1]*10 << " " << com0[2]*10 << endl;
 wfile.close();
}

