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
    vector<double> xyz;
    vector<double> d_com_dx;
    vector<double> d_com_dy;
    vector<double> d_com_dz;
    
  public:
    Com();
    void init(unsigned N_ligand, vector<double> Masses_ligand);
    void calculate_com(vector<vector<double>> ligand_xyz);
};
#endif