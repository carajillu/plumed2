#include <vector>
#include "tools/PDB.h"
#include <armadillo>

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

  vector<double> rx;
  vector<double> ry;
  vector<double> rz;

  vector<double> r;
  vector<double> dr_dx;
  vector<double> dr_dy;
  vector<double> dr_dz;

  vector<double> Soff_r;
  double total_Soff;
  vector<double> dSoff_r_dx;
  vector<double> dSoff_r_dy;
  vector<double> dSoff_r_dz;

  //coordinates
  vector<double> xyz;
  vector<double> centroid;
  vector<double> centroid0;

  //for probe movement
  arma::mat arma_xyz;
  arma::mat atomcoords_0;
  arma::mat atomcoords;
  arma::mat weights;
  arma::mat wCov; // weighted covariance matrix
  arma::mat U;
  arma::vec s;
  arma::mat V;
  arma::mat R; //rotation matrix

 public:
    Probe(double Rprobe, double Mind_slope, double Mind_intercept, double CCMin, double CCMax,double DeltaCC, double DMin, double DeltaD, unsigned n_atoms);
    
    void place_probe(double x, double y, double z);

    void calc_centroid(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z, unsigned n_atoms);
    void kabsch(unsigned step, vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z, unsigned n_atoms, vector<double> masses, double total_mass);
    void move_probe();

    void calculate_r(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z, unsigned n_atoms);
    void calculate_Soff_r(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z, unsigned n_atoms);

    void print_probe_movement(int id, int step, vector<PLMD::AtomNumber> atoms, unsigned n_atoms, double ref_x, double ref_y, double ref_z);
    void print_probe_xyz(int id, int step);

};
#endif
