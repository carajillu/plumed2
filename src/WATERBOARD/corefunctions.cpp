#include <iostream>
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