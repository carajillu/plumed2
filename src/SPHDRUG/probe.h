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
  unsigned n_atoms;
  double mind_slope; //slope of the mind linear implementation
  double mind_intercept; //intercept of the mind linear implementation
  double CCmin; // mind below which an atom is considered to be clashing with the probe 
  double CCmax; // distance above which an atom is considered to be too far away from the probe*
  double deltaCC; // interval over which contact terms are turned on and off
  double Dmin; // packing factor below which depth term equals 0
  double deltaD; // interval over which depth term turns from 0 to 1
  double Kpert;
  
  //stuff

  vector<double> rx;
  vector<double> ry;
  vector<double> rz;

  vector<double> r;
  vector<double> dr_dx;
  vector<double> dr_dy;
  vector<double> dr_dz;
  void calculate_r(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z);
  //in case the probe gets too far
  double min_r; 
  unsigned j_min_r;

  vector<double> Soff_r;
  double total_Soff;
  vector<double> dSoff_r_dx;
  vector<double> dSoff_r_dy;
  vector<double> dSoff_r_dz;
  void calculate_Soff_r();

  vector<double> Son_r;
  double total_Son;
  vector<double> dSon_r_dx;
  vector<double> dSon_r_dy;
  vector<double> dSon_r_dz;
  void calculate_Son_r();

  double mind;
  vector<double> dmind_dx;
  vector<double> dmind_dy;
  vector<double> dmind_dz;
  void calculate_mind();

  double CC;
  double dCC_dr;
  vector<double> dCC_dx;
  vector<double> dCC_dy;
  vector<double> dCC_dz;
  void calculate_CC();

  double D;
  vector<double> dD_dx;
  vector<double> dD_dy;
  vector<double> dD_dz;
  void calculate_D();
  
  double H;
  vector<double> dH_dx;
  vector<double> dH_dy;
  vector<double> dH_dz;
  void calculate_H();

  //coordinates
  vector<double> xyz;
  vector<double> centroid;
  vector<double> centroid0;

  //for probe movement
  arma::mat arma_xyz;
  arma::mat atomcoords_0;
  arma::mat atomcoords;
  arma::mat wCov; // weighted covariance matrix
  arma::mat weights; 
  arma::mat U;
  arma::vec s;
  arma::mat V;
  arma::mat R; //rotation matrix
  void kabsch();

  // for probe perturbation
  vector<double> xyz_pert;
  vector<double> xyz0;
  unsigned ptries;
  void calc_pert();

 public:
    Probe(double Mind_slope, double Mind_intercept, double CCMin, double CCMax,double DeltaCC, double DMin, double DeltaD, unsigned n_atoms, double kpert);
    
    void place_probe(double x, double y, double z);
    void perturb_probe(unsigned step, vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z);

    
    void move_probe(unsigned step, vector<double> atoms_x,vector<double> atoms_y, vector<double> atoms_z);

    double activity;
    double activity_cum; // cummulative activity over PERTSTRIDE steps
    double activity_old; // cummulative activity over the last period 
    double Dcum; //cumulative depth over PERTSTRIDE steps
    double Dold; //cummulative depth over the last period
    double pert_accepted;
    double pert_rejected;
    double pert_acceptance;
    string accepted;
    vector<double> d_activity_dx;
    vector<double> d_activity_dy;
    vector<double> d_activity_dz;
    void calculate_activity(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z);

    void print_probe_movement(int id, int step, vector<PLMD::AtomNumber> atoms, unsigned n_atoms, double ref_x, double ref_y, double ref_z);
    void print_probe_xyz(int id, int step);

};
#endif
