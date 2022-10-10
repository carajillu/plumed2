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
  double Rmin; // mind below which an atom is considered to be clashing with the probe 
  double deltaRmin; // interval over which contact terms are turned on and off
  double Rmax; // distance above which an atom is considered to be too far away from the probe*
  double deltaRmax; // interval over which contact terms are turned on and off
  double Cmin; // packing factor below which depth term equals 0
  double deltaC; // interval over which depth term turns from 0 to 1
  double Pmin; // packing factor below which depth term equals 0
  double deltaP; // interval over which depth term turns from 0 to 1
  double Kpert;
  double theta;
  
  //stuff

  unsigned probe_id;
  unsigned init_j; //index j of atom on which the probe is centered at the beginning
  bool dxcalc; // avoid derivative calculation when perturbing probe

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

  //Enclosure score
  vector<double> enclosure;
  double total_enclosure;
  vector<double> d_enclosure_dx;
  vector<double> d_enclosure_dy;
  vector<double> d_enclosure_dz;
  void calculate_enclosure();

  double P;
  vector<double> dP_dx;
  vector<double> dP_dy;
  vector<double> dP_dz;
  //P=S_on(total_enclosure)
  void calculate_P(); 

  vector<double> clash;
  double total_clash;
  vector<double> d_clash_dx;
  vector<double> d_clash_dy;
  vector<double> d_clash_dz;
  void calculate_clash();

  double C;
  vector<double> dC_dx;
  vector<double> dC_dy;
  vector<double> dC_dz;
  //C=S_off(total_clash)
  void calculate_C();


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
    Probe(unsigned Probe_id, 
          double RMin, double DeltaRmin, 
          double RMax, double DeltaRmax, 
          double phimin, double deltaphi, 
          double psimin, double deltapsi, 
          unsigned N_atoms, double kpert, 
          unsigned init_j);
    
    void place_probe(double x, double y, double z);
    void perturb_probe(unsigned step, vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z);

    
    void move_probe(unsigned step, vector<double> atoms_x,vector<double> atoms_y, vector<double> atoms_z);

    double activity;
    double activity_cum; // cummulative activity over PERTSTRIDE steps
    double activity_old; // cummulative activity over the last period 
    double r_target; //distance between the probe and the target region (if not specified, r_target=INFINITY)
    vector<double> d_activity_dx;
    vector<double> d_activity_dy;
    vector<double> d_activity_dz;
    void calculate_activity(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z);

    void print_probe_movement(int id, int step, vector<PLMD::AtomNumber> atoms, unsigned n_atoms, vector<double> target_xyz);
    void print_probe_xyz(int id, int step);
};
#endif
