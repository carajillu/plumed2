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
  double CCmin; // mind below which an atom is considered to be clashing with the probe 
  double CCmax; // distance above which an atom is considered to be too far away from the probe*
  double deltaCC; // interval over which contact terms are turned on and off
  double Phimin; // packing factor below which depth term equals 0
  double deltaPhi; // interval over which depth term turns from 0 to 1
  double Kpert;
  double theta;
  
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
  vector<double> dSoff_r_dx;
  vector<double> dSoff_r_dy;
  vector<double> dSoff_r_dz;
  void calculate_Soff_r();

  vector<double> Son_r;
  vector<double> dSon_r_dx;
  vector<double> dSon_r_dy;
  vector<double> dSon_r_dz;
  void calculate_Son_r();

  double Phi;
  vector<double> dPhi_dx;
  vector<double> dPhi_dy;
  vector<double> dPhi_dz;
  void calculate_Phi();


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
    Probe(double CCMin, double CCMax, double DeltaCC, double phimin, double deltaphi, unsigned N_atoms, double kpert);
    
    void place_probe(double x, double y, double z);
    void perturb_probe(unsigned step, vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z);

    
    void move_probe(unsigned step, vector<double> atoms_x,vector<double> atoms_y, vector<double> atoms_z);

    double activity;
    double activity_cum; // cummulative activity over PERTSTRIDE steps
    double activity_old; // cummulative activity over the last period 
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
