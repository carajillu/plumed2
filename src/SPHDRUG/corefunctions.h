#include <iostream>
#include <vector>
#include "tools/PDB.h"

using namespace std;

#ifndef COREFUNCTIONS_h_
#define COREFUNCTIONS_h_

namespace COREFUNCTIONS
{
double m_v(double v, double v0, double delta);
double dm_dv(double delta);

double Son_m(double m, double k);
double dSon_dm(double m, double k);

double Soff_m(double m, double k);

double dSoff_dm(double m, double k);
}
#endif