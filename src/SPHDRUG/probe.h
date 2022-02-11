#include <vector>
#include "tools/PDB.h"

#define max_atoms 10000

using namespace std;

#ifndef probe_h_
#define probe_h_

class Probe
{
 private:
  //parameters
  double rprobe; // radius of each spherical probe
  double mind_slope; //slope of the mind linear implementation
  double mind_intercept; //intercept of the mind linear implementation
  double CCmin; // mind below which an atom is considered to be clashing with the probe 
  double CCmax; // distance above which an atom is considered to be too far away from the probe*
  double deltaCC; // interval over which contact terms are turned on and off
  double Pmax; // number of atoms surrounding the probe for it to be considered completely packed
  double Dmin; // packing factor below which depth term equals 0
  double deltaD; // interval over which depth term turns from 0 to 1
  
  //stuff

  double rx[max_atoms];
  double ry[max_atoms];
  double rz[max_atoms];

  double r[max_atoms];
  double dr_dx[max_atoms];
  double dr_dy[max_atoms];
  double dr_dz[max_atoms];

  double Soff_r[max_atoms];
  double total_Soff;
  double dSoff_r_dx[max_atoms];
  double dSoff_r_dy[max_atoms];
  double dSoff_r_dz[max_atoms];

  //coordinates
  double xyz[3];
  double xyz0[3];
  double centroid[3];
  double centroid0[3];
  double com[3];
  double com0[3];
  double com_bckp[3];
  double centroid_bckp[3];

  //for probe movement
  double com_x;
  double com_y;
  double com_z;
  double centroid_x;
  double centroid_y;
  double centroid_z;
  double ref_xlist[2][3]; //COM and centroid
  double mov_xlist[2][3]; //COM and centroid
  //double mov_com[3];
  //double move_to_ref[3];
  double comcen[3]; // centroid - com
  double comcen0[3]; // centroid - com
  double comcen_norm;
  double comcen0_norm;
  double cross[3];
  double deltacentre[3];
  double rotmat[3][3];
  double rmsd;
  double mov_mass_tot;

  double atoms_x0[max_atoms];
  double atoms_y0[max_atoms];
  double atoms_z0[max_atoms];

 public:
    Probe(double Rprobe, double Mind_slope, double Mind_intercept, double CCMin, double CCMax,double DeltaCC, double DMin, double DeltaD);
    
    void place_probe(double x, double y, double z);
    void calc_centroid(double* atoms_x, double* atoms_y, double* atoms_z, unsigned n_atoms);
    void move_probe(unsigned step, double* atoms_x, double* atoms_y, double* atoms_z, unsigned n_atoms, double* masses, double total_mass);
    
    void calculate_r(double* atoms_x, double* atoms_y, double* atoms_z, unsigned n_atoms);
    void calculate_Soff_r(double* atoms_x, double* atoms_y, double* atoms_z, unsigned n_atoms);

    void print_probe_movement(int id, int step, vector<PLMD::AtomNumber> atoms, unsigned n_atoms, double ref_x, double ref_y, double ref_z);
    void print_probe_xyz(int id, int step);

};
#endif
