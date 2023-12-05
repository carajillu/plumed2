#include <iostream>
#include <cmath>
#include <vector>

#include "corefunctions.h"

using namespace std;

vector<double> COREFUNCTIONS::calculate_com(vector<vector<double>> atoms_xyz, unsigned n_ligand, vector<double> masses_ligand,double ligand_total_mass)
{
 vector<double> com_xyz(3);
 for (unsigned j=0; j<n_ligand; j++)
 {
   com_xyz[0]+=atoms_xyz[j][0]*masses_ligand[j];
   com_xyz[1]+=atoms_xyz[j][1]*masses_ligand[j];
   com_xyz[2]+=atoms_xyz[j][2]*masses_ligand[j];
 } 
 com_xyz[0]/=ligand_total_mass;
 com_xyz[1]/=ligand_total_mass;
 com_xyz[2]/=ligand_total_mass;
 return com_xyz;
}

static void COREFUNCTIONS::calculate_rx(vector<double> &rx, 
                                        vector<double> &com, vector<vector<double>> &atoms_xyz,
                                        unsigned n_ligand, unsigned n_atoms, unsigned x_pos)
{
 #pragma omp parallel for
 for (unsigned j=n_ligand; j<n_atoms; j++)
 {
  rx[j]=(com[x_pos]-atoms_xyz[j][x_pos]);
 }
 return;
}

static void COREFUNCTIONS::calculate_r(vector<double> &r,
                                       vector<double> &rx, vector<double> &ry,vector<double> &rz,
                                       unsigned n_ligand, unsigned n_atoms)
{
  #pragma omp parallel for
  for (unsigned j=n_ligand; j<n_atoms; j++)
  {
   r[j]=sqrt(pow(rx,2)+pow(ry,2)+pow(rz,2));
  }
  return;
}

static void calculate_dr_dx(vector<double> &dr_dx, 
                            vector<double> &r, vector<double> &rx,
                            unsigned n_ligand, unsigned n_atoms, unsigned x_pos)
{
  #pragma omp parallel for
  for (unsigned j=n_ligand; j<n_atoms; j++)
  {
   dr_dx[j]=rx[j]/r[j];
  }
  return;
}