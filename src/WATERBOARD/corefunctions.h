#include <iostream>
#include <vector>
#include "tools/PDB.h"

using namespace std;

#ifndef COREFUNCTIONS_h_
#define COREFUNCTIONS_h_

namespace COREFUNCTIONS
{
 vector<double> calculate_com(vector<vector<double>>, unsigned n_ligand, vector<double> masses_ligand,double ligand_total_mass);

 static void calculate_rx(vector<double> &rx, 
                          vector<double> &com, vector<vector<double>> &atoms_xyz,
                          unsigned n_ligand, unsigned n_atoms, unsigned x_pos);

 static void calculate_r(vector<double> &r,
                         vector<double> &rx,vector<double> &ry,vector<double> &rz,
                         unsigned n_ligand, unsigned n_atoms);

 static void calculate_dr_dx(vector<double> &dr_dx, 
                             vector<double> &r, vector<double> &rx,
                             unsigned n_ligand, unsigned n_atoms);
}
#endif