#include <iostream>
#include <vector>
#include "com.h"

using namespace std;

Com::Com(unsigned N_ligand, vector<double> Masses_ligand)
{
 n_ligand=N_ligand;
 masses_ligand=Masses_ligand;
 //#pragma omp parallel for reduction(+:total_mass)
 for (unsigned j=0;j<n_ligand;j++)
 {
  cout << "Mass of atom:" << j << " " << masses_ligand[j] << endl;
  total_mass+=masses_ligand[j];
 }
 cout << "Total Mass: " << total_mass << endl;

 xyz=vector<double>(3,0);
 
 //there must be a better way to do this since the 3 derivatives are the same
 d_com_dx=vector<double>(n_ligand,0);
 d_com_dy=vector<double>(n_ligand,0);
 d_com_dz=vector<double>(n_ligand,0);
 for (unsigned j=0; j<n_ligand; j++)
 {
  d_com_dx[j]=masses_ligand[j]/total_mass;
  d_com_dy[j]=masses_ligand[j]/total_mass;
  d_com_dz[j]=masses_ligand[j]/total_mass;
 }
}

void Com::calculate_com(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z)
{
 xyz[0]=0;
 xyz[1]=0;
 xyz[2]=0;
 //#pragma omp parallel for reduction(+:com_x,com_y,com_z) //parallelise if too slow
 for (unsigned j=0;j<n_ligand;j++)
 {
  xyz[0]+=atoms_x[j]*masses_ligand[j];
  xyz[1]+=atoms_y[j]*masses_ligand[j];
  xyz[2]+=atoms_z[j]*masses_ligand[j];
 }
 xyz[0]/=total_mass;
 xyz[1]/=total_mass;
 xyz[2]/=total_mass;
}
