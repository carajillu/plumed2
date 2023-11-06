#include <vector>

using namespace std;

#ifndef COM_h_
#define COM_h_

class Com
{
  private:
    unsigned n_ligand;
    vector<double> masses_ligand;
    double total_mass;
    

    vector<double> d_com_dx;
    vector<double> d_com_dy;
    vector<double> d_com_dz;
  public:
    Com(unsigned N_ligand, vector<double> Masses_ligand);
    vector<double> xyz;
    void calculate_com(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z);
};
#endif