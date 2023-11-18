#include <iostream>
#include <vector>
#include "tools/PDB.h"

using namespace std;

#ifndef COREFUNCTIONS_h_
#define COREFUNCTIONS_h_

namespace COREFUNCTIONS
{
 vector<double> calculate_com(vector<vector<double>>, unsigned n_ligand, vector<double> masses_ligand,double ligand_total_mass);
 vector<double> calculate_r();
 vector<double> calculate_dr();
}
#endif