#include <iostream>
#include "corefunctions.h"
#include <random>

using namespace std;

double COREFUNCTIONS::m_v(double v, double v0, double delta)
{
    double m=(v-v0)/delta;
    return m;
}

double COREFUNCTIONS::dm_dv(double delta)
{
    return 1/delta;
}

double COREFUNCTIONS::Son_m(double m, double k)
{
    double S_on=0;
    if (m<=0) 
        S_on=0;
    else if ((m>0) and (m<1))
        S_on=k*(3*pow(m,4)-2*pow(m,6));
    else 
        S_on=k;
    return S_on;
}

double COREFUNCTIONS::dSon_dm(double m, double k)
{
    double dS_on=0;
    if (m<=0 or m>=1) 
        dS_on=0;
    else
        dS_on=k*(12*(pow(m,3)-pow(m,5)));
    return dS_on;
}

double COREFUNCTIONS::Soff_m(double m, double k)
{
    double S_off=0;
    if (m<=0) 
        S_off=k;
    else if ((m>0) and (m<1))
        S_off=k*(3*pow((m-1),4)-2*pow((m-1),6));
    else S_off=0;
    return S_off;
}

double COREFUNCTIONS::dSoff_dm(double m, double k)
{
    double dS_off=0;
    if (m<=0 or m>=1) 
        dS_off=0;
    else
        dS_off=k*(12*(pow((m-1),3)-pow((m-1),5)));
    return dS_off;
}

double COREFUNCTIONS::random_double(double bmin, double bmax)
{
  random_device rd;  // only used once to initialise (seed) engine
  mt19937 rng(rd()); // random-number engine used (Mersenne-Twister in this case)
  uniform_real_distribution<double> uni(-1, 1); // guaranteed unbiased
  auto random_double = uni(rng);
  return random_double;
}