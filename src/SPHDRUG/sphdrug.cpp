/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2019 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"

#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <armadillo>
#include <omp.h>

// CV modules
#include "probe.h"

using namespace std;
using namespace std::chrono;

namespace PLMD
{
  namespace colvar
  {

    /*+PLUMEDOC COLVAR TEMPLATE
    Add CV info
    +ENDPLUMEDOC*/

    class Sphdrug : public Colvar
    {
      // Execution control variables
      int nthreads;     // number of available OMP threads
      int ndev;         // number of available OMP accelerators
      bool performance; // print execution time
      // MD control variables
      bool pbc;
      // CV control variables
      bool nodxfix;
      bool noupdate;
      // Variables necessary to check results
      bool target;
      vector<PLMD::AtomNumber> atoms_target; // indices of the atoms defining the target pocket
      unsigned n_target;                     // number of atoms used in TARGET
      vector<unsigned> target_j;             // Indices of atoms_target in getPositions()
      // Parameters
      double rprobe;         // radius of each spherical probe
      double mind_slope;     // slope of the mind linear implementation
      double mind_intercept; // intercept of the mind linear implementation
      double CCmin;          // mind below which an atom is considered to be clashing with the probe
      double CCmax;          // distance above which an atom is considered to be too far away from the probe*
      double deltaCC;        // interval over which contact terms are turned on and off
      double Dmin;           // packing factor below which depth term equals 0
      double deltaD;         // interval over which depth term turns from 0 to 1

      // Set up of CV
      vector<PLMD::AtomNumber> atoms; // indices of atoms supplied to the CV (starts at 1)
      unsigned n_atoms;               // number of atoms supplied to the CV
      vector<double> masses;
      double total_mass;
      vector<double> atoms_x;
      vector<double> atoms_y;
      vector<double> atoms_z;
      unsigned step;

      vector<PLMD::AtomNumber> atoms_init; // Indices of the atoms in which the probes will be initially centered
      unsigned n_init;                     // number of atoms used in ATOMS_INIT
      vector<unsigned> init_j;             // Indices of atoms_init in getPositions()

      vector<Probe> probes; // This will contain all the spherical probes
      unsigned nprobes;     // number of spherical probes to use

      // Output control variables
      unsigned probestride; // stride to print information for post-processing the probe coordinates

      // Calculation of CV and its derivatives

      double sphdrug;
      vector<double> d_Sphdrug_dx;
      vector<double> d_Sphdrug_dy;
      vector<double> d_Sphdrug_dz;

      // Correction of derivatives
      double sum_d_dx;
      double sum_d_dy;
      double sum_d_dz;
      double sum_t_dx;
      double sum_t_dy;
      double sum_t_dz;

      arma::vec L;
      unsigned nrows;
      unsigned ncols;
      arma::mat A;
      arma::mat Aplus;
      arma::vec P;
      vector<double> sum_P;
      vector<double> sum_rcrossP;

    public:
      explicit Sphdrug(const ActionOptions &);
      // active methods:
      void calculate() override;
      void reset();
      void correct_derivatives();
      void print_protein();
      static void registerKeywords(Keywords &keys);
    };

    PLUMED_REGISTER_ACTION(Sphdrug, "SPHDRUG")

    void Sphdrug::registerKeywords(Keywords &keys)
    {
      Colvar::registerKeywords(keys);
      keys.addFlag("DEBUG", false, "Running in debug mode");
      keys.addFlag("NOUPDATE", false, "skip probe update");
      keys.addFlag("NODXFIX", false, "skip derivative correction");
      keys.addFlag("PERFORMANCE", false, "measure execution time");
      keys.add("atoms", "ATOMS", "Atoms to include in druggability calculations (start at 1)");
      keys.add("atoms", "ATOMS_INIT", "Atoms in which the probes will be initially centered.");
      keys.add("atoms", "TARGET", "Atoms defining the target pocket (not necessarily among ATOMS)");
      keys.add("optional", "NPROBES", "Number of probes to use");
      keys.add("optional", "RPROBE", "Radius of every probe in nm");
      keys.add("optional", "PROBESTRIDE", "Print probe coordinates info every PROBESTRIDE steps");
      keys.add("optional", "CCMIN", "");
      keys.add("optional", "CCMAX", "");
      keys.add("optional", "DELTACC", "");
      keys.add("optional", "MINDSLOPE", "");
      keys.add("optional", "MINDINTERCEPT", "");
      keys.add("optional", "DMIN", "");
      keys.add("optional", "DELTAD", "");
    }

    Sphdrug::Sphdrug(const ActionOptions &ao) : PLUMED_COLVAR_INIT(ao),
                                                pbc(true),
                                                noupdate(false),
                                                nodxfix(false),
                                                performance(false),
                                                target(true)
    {
/*
Initialising openMP threads.
This does not seem to be affected by the environment variable $PLUMED_NUM_THREADS
*/
#pragma omp parallel
      nthreads = omp_get_num_threads();
      ndev = omp_get_num_devices();
      cout << "------------ Available Computing Resources -------------" << endl;
      cout << "Sphdrug initialised with " << nthreads << " OMP threads " << endl;
      cout << "and " << ndev << " OMP compatible accelerators (not currently used)" << endl;

      addValueWithDerivatives();
      setNotPeriodic();

      bool nopbc = !pbc;
      parseFlag("NOPBC", nopbc);
      pbc = !nopbc;

      parseFlag("NOUPDATE", noupdate);
      parseFlag("NODXFIX", nodxfix);
      parseFlag("PERFORMANCE", performance);

      parseAtomList("ATOMS", atoms);
      parseAtomList("ATOMS_INIT", atoms_init);
      parseAtomList("TARGET", atoms_target);

      n_atoms = atoms.size();

      n_init = atoms_init.size();
      for (unsigned j = 0; j < n_init; j++)
      {
        atoms.push_back(atoms_init[j]);
        init_j.push_back(n_atoms + j);
      }

      if (atoms_target.size() == 0)
      {
        target = false;
      }
      else
      {
        n_target = atoms_target.size();
        for (unsigned j = 0; j < n_target; j++)
        {
          atoms.push_back(atoms_target[j]);
          target_j.push_back(n_atoms + n_init + j);
        }
      }

      cout << "Requesting " << n_atoms << " atoms" << endl;
      requestAtoms(atoms);
      cout << "--------- Initialising Sphdrug Collective Variable -----------" << endl;

      parse("NPROBES", nprobes);
      if (!nprobes)
      {
        nprobes = 1;
      }

      if (nprobes != atoms_init.size() and atoms_init.size() > 0)
      {
        cout << "Overriding NPROBES with the length of ATOMS_INIT" << endl;
        nprobes = atoms_init.size();
      }

      if (atoms_init.size() == 0)
      {
        for (unsigned i = 0; i < nprobes; i++)
        {
          random_device rd;                                   // only used once to initialise (seed) engine
          mt19937 rng(rd());                                  // random-number engine used (Mersenne-Twister in this case)
          uniform_int_distribution<unsigned> uni(0, n_atoms); // guaranteed unbiased
          auto random_integer = uni(rng);
          init_j.push_back(random_integer);
        }
      }

      cout << "Using " << nprobes << " spherical probe(s) with the following parameters:" << endl;

      // Attributes of the probe object (kept private)
      parse("RPROBE", rprobe);
      if (!rprobe)
        rprobe = 0.3;
      cout << "Radius = " << rprobe << " nm" << endl;

      parse("CCMIN", CCmin);
      if (!CCmin)
        CCmin = 0.2;
      cout << "CCmin = " << CCmin << " nm" << endl;

      parse("CCMAX", CCmax);
      if (!CCmax)
        CCmax = 0.5;
      cout << "CCmax = " << CCmax << " nm" << endl;

      parse("DELTACC", deltaCC);
      if (!deltaCC)
        deltaCC = 0.05;
      cout << "deltaCC = " << deltaCC << " nm" << endl;

      parse("MINDSLOPE", mind_slope);
      if (!mind_slope)
        mind_slope = 1.227666; // obtained from generating 10000 random points in VHL's crystal structure
      cout << "MINDSLOPE = " << mind_slope << endl;

      parse("MINDINTERCEPT", mind_intercept);
      if (!mind_intercept)
        mind_intercept = -0.089870; // obtained from generating 10000 random points in VHL's crystal structure
      cout << "MINDINTERCEPT = " << mind_intercept << " nm" << endl;

      parse("DMIN", Dmin);
      if (!Dmin)
        Dmin = 10; // obtained from generating 10000 random points in VHL's crystal structure
      cout << "DMIN = " << Dmin << endl;

      parse("DELTAD", deltaD);
      if (!deltaD)
        deltaD = 15; // obtained from generating 10000 random points in VHL's crystal structure
      cout << "DELTAD = " << deltaD << endl;

      for (unsigned i = 0; i < nprobes; i++)
      {

        probes.push_back(Probe(rprobe, mind_slope, mind_intercept, CCmin, CCmax, deltaCC, Dmin, deltaD, n_atoms));
        cout << "Probe " << i << " initialised, centered on atom: " << to_string(atoms[init_j[i]].serial()) << endl;
      }

      // parameters used to control output
      parse("PROBESTRIDE", probestride);
      if (!probestride)
        probestride = 1;
      cout << "Information to post-process probe coordinates will be printed every " << probestride << " steps" << endl
           << endl;

      checkRead();

      // Allocate space for atom coordinates and masses

      atoms_x = vector<double>(n_atoms, 0);
      atoms_y = vector<double>(n_atoms, 0);
      atoms_z = vector<double>(n_atoms, 0);
      masses = vector<double>(n_atoms, 0);

      cout << "---------Initialisng Sphdrug and its derivatives---------" << endl;
      sphdrug = 0;
      d_Sphdrug_dx = vector<double>(n_atoms, 0);
      d_Sphdrug_dy = vector<double>(n_atoms, 0);
      d_Sphdrug_dz = vector<double>(n_atoms, 0);
      if (!nodxfix)
      {
        cout << "---------Initialisng correction of Sphdrug derivatives---------" << endl;

        // L=vector<double>(6,0); //sums of derivatives and sums torques in each direction
        nrows = 6;
        ncols = 3 * n_atoms;
        A = arma::mat(nrows, ncols);
        Aplus = arma::mat(nrows, ncols);
        L = arma::vec(nrows);
        P = arma::vec(ncols);
        sum_P = vector<double>(3, 0);
        sum_rcrossP = vector<double>(3, 0);
      }
      else
      {
        cout << "Sphdrug derivatives are not going to be corrected" << endl;
        cout << "Use the NODXFIX flag with care, as this means that" << endl;
        cout << "the sum of forces in the system will not be zero" << endl;
      }

      cout << "--------- Initialisation complete -----------" << endl;
    }

    // reset Sphdrug and derivatives to 0
    void Sphdrug::reset()
    {
      sphdrug = 0;
      fill(d_Sphdrug_dx.begin(), d_Sphdrug_dx.end(), 0);
      fill(d_Sphdrug_dy.begin(), d_Sphdrug_dy.end(), 0);
      fill(d_Sphdrug_dz.begin(), d_Sphdrug_dz.end(), 0);

      sum_d_dx = 0;
      sum_d_dy = 0;
      sum_d_dz = 0;
      sum_t_dx = 0;
      sum_t_dy = 0;
      sum_t_dz = 0;
      fill(L.begin(), L.end(), 0);
      fill(P.begin(), P.end(), 0);
      fill(sum_P.begin(), sum_P.end(), 0);
      fill(sum_rcrossP.begin(), sum_rcrossP.end(), 0);
    }

    void Sphdrug::correct_derivatives()
    {
      // auto point0=high_resolution_clock::now();
      // step 0: calculate sums of derivatives and sums of torques in each direction
      for (unsigned j = 0; j < n_atoms; j++)
      {
        sum_d_dx += d_Sphdrug_dx[j];
        sum_d_dy += d_Sphdrug_dy[j];
        sum_d_dz += d_Sphdrug_dz[j];

        sum_t_dx += atoms_y[j] * d_Sphdrug_dz[j] - atoms_z[j] * d_Sphdrug_dy[j];
        sum_t_dy += atoms_z[j] * d_Sphdrug_dx[j] - atoms_x[j] * d_Sphdrug_dz[j];
        sum_t_dz += atoms_x[j] * d_Sphdrug_dy[j] - atoms_y[j] * d_Sphdrug_dx[j];
      }
      L[0] = -sum_d_dx;
      L[1] = -sum_d_dy;
      L[2] = -sum_d_dz;
      L[3] = -sum_t_dx;
      L[4] = -sum_t_dy;
      L[5] = -sum_t_dz;

// only for debugging
// cout << "Before: " << L[0] << " " << L[1] << " " << L[2] << " " << L[3] << " "<< L[4] << " "<< L[5] << endl;

// step2 jedi.cpp
// auto point1=high_resolution_clock::now();
#pragma omp parallel for
      for (unsigned j = 0; j < ncols; j++)
      {
        if (j < n_atoms)
        {
          A.row(0).col(j) = 1.0;
          A.row(1).col(j) = 0.0;
          A.row(2).col(j) = 0.0;
          A.row(3).col(j) = 0.0;
          A.row(4).col(j) = atoms_z[j];
          A.row(5).col(j) = -atoms_y[j];
        }
        else if (j < 2 * n_atoms)
        {
          A.row(0).col(j) = 0.0;
          A.row(1).col(j) = 1.0;
          A.row(2).col(j) = 0.0;
          A.row(3).col(j) = atoms_z[j - n_atoms];
          A.row(4).col(j) = 0.0;
          A.row(5).col(j) = atoms_x[j - n_atoms];
        }
        else
        {
          A.row(0).col(j) = 0.0;
          A.row(1).col(j) = 0.0;
          A.row(2).col(j) = 1.0;
          A.row(3).col(j) = atoms_y[j - 2 * n_atoms];
          A.row(4).col(j) = -atoms_x[j - 2 * n_atoms];
          A.row(5).col(j) = 0.0;
        }
      }

      // auto point2=high_resolution_clock::now();
      // step3 jedi.cpp
      Aplus = arma::pinv(A);

      // auto point3=high_resolution_clock::now();
      // step4 jedi.cpp
      P = Aplus * L;

      // auto point4=high_resolution_clock::now();
      for (unsigned j = 0; j < n_atoms; j++)
      {
        sum_P[0] += P[j + 0 * n_atoms];
        sum_P[1] += P[j + 1 * n_atoms];
        sum_P[2] += P[j + 2 * n_atoms];

        sum_rcrossP[0] += atoms_y[j] * P[j + 2 * n_atoms] - atoms_z[j] * P[j + 1 * n_atoms];
        sum_rcrossP[0] += atoms_z[j] * P[j + 0 * n_atoms] - atoms_x[j] * P[j + 2 * n_atoms];
        sum_rcrossP[0] += atoms_x[j] * P[j + 1 * n_atoms] - atoms_y[j] * P[j + 0 * n_atoms];
      }

      // step5 jedi.cpp
      for (unsigned j = 0; j < n_atoms; j++)
      {
        d_Sphdrug_dx[j] += P[j + 0 * n_atoms];
        d_Sphdrug_dy[j] += P[j + 1 * n_atoms];
        d_Sphdrug_dz[j] += P[j + 2 * n_atoms];
      }
      // auto point5=high_resolution_clock::now();

      // Only for debugging
      /*
      sum_d_dx=0;
      sum_d_dy=0;
      sum_d_dz=0;
      sum_t_dx=0;
      sum_t_dy=0;
      sum_t_dz=0;

      for (unsigned j=0;j<n_atoms;j++)
      {
       sum_d_dx+=d_Sphdrug_dx[j];
       sum_d_dy+=d_Sphdrug_dy[j];
       sum_d_dz+=d_Sphdrug_dz[j];

       sum_t_dx+=atom_crd[j][1]*d_Sphdrug_dz[j]-atom_crd[j][2]*d_Sphdrug_dy[j];
       sum_t_dy+=atom_crd[j][2]*d_Sphdrug_dx[j]-atom_crd[j][0]*d_Sphdrug_dz[j];
       sum_t_dz+=atom_crd[j][0]*d_Sphdrug_dy[j]-atom_crd[j][1]*d_Sphdrug_dx[j];
      }
      L[0]=-sum_d_dx;
      L[1]=-sum_d_dy;
      L[2]=-sum_d_dz;
      L[3]=-sum_t_dx;
      L[4]=-sum_t_dy;
      L[5]=-sum_t_dz;

      cout << "After: " << L[0] << " " << L[1] << " " << L[2] << " " << L[3] << " "<< L[4] << " "<< L[5] << endl;

      for (unsigned k=0; k<P.size();k++)
      {
        cout << P[k] << endl;
      }
      */
      /*
      auto duration0 = duration_cast<microseconds>(point1 - point0);
      auto duration1 = duration_cast<microseconds>(point2 - point1);
      auto duration2 = duration_cast<microseconds>(point3 - point2);
      auto duration3 = duration_cast<microseconds>(point4 - point3);
      auto duration4 = duration_cast<microseconds>(point5 - point4);
      auto duration5 = duration_cast<microseconds>(point5 - point0);
      cout << "Duration L = " << duration0.count() << " microseconds" << endl;
      cout << "Duration A = " << duration1.count() << " microseconds" << endl;
      cout << "Duration Aplus = " << duration2.count() << " microseconds" << endl;
      cout << "Duration P = " << duration3.count() << " microseconds" << endl;
      cout << "Duration Correction = " << duration4.count() << " microseconds" << endl;
      cout << "Duration Total = " << duration5.count() << " microseconds" << endl;

      exit(0);
      */
    }

    void Sphdrug::print_protein()
    {
     string filename = "protein.xyz";
     ofstream wfile;
     if (step==0)
     {
      wfile.open(filename.c_str());
     }
     else
     {
      wfile.open(filename.c_str(),std::ios_base::app);
     }
     wfile << n_atoms << endl;
     wfile << "Step  "<< to_string(step) << endl;
     for (unsigned j=0; j<n_atoms;j++)
     {
      wfile << atoms[j].serial() << " " << std::fixed << std::setprecision(5) << atoms_x[j]*10 << " " << atoms_y[j]*10 << " " << atoms_z[j]*10 << endl;  
     } 
     wfile.close();
    }

    // calculator
    void Sphdrug::calculate()
    {

      auto start_psi = high_resolution_clock::now();
      if (pbc)
        makeWhole();
      reset();

      step = getStep();
      //#pragma omp parallel for //?
      for (unsigned j = 0; j < n_atoms; j++)
      {
        atoms_x[j] = getPosition(j)[0];
        atoms_y[j] = getPosition(j)[1];
        atoms_z[j] = getPosition(j)[2];
        if (step > 0) // masses are the same throughout the simulation
          continue;
        // masses[j]=getMass(j);
        masses[j] = 1;
        total_mass += masses[j];
      }

      #pragma omp parallel for
      for (unsigned i = 0; i < nprobes; i++)
      {
        // set up stuff at step 0
        if (step == 0)
        {
          double x = getPosition(init_j[i])[0];
          double y = getPosition(init_j[i])[1];
          double z = getPosition(init_j[i])[2];
          probes[i].place_probe(x, y, z);
          probes[i].calculate_r(atoms_x, atoms_y, atoms_z, n_atoms);
          probes[i].calculate_Soff_r(atoms_x, atoms_y, atoms_z, n_atoms);
        }
        // Update probe coordinates
        probes[i].calc_centroid(atoms_x, atoms_y, atoms_z, n_atoms);
        probes[i].kabsch(step, atoms_x, atoms_y, atoms_z, n_atoms, masses, total_mass);
        probes[i].move_probe();
        probes[i].calculate_r(atoms_x, atoms_y, atoms_z, n_atoms);
        probes[i].calculate_Soff_r(atoms_x, atoms_y, atoms_z, n_atoms);
      }
      
      //print output for post_processing
      if (step % probestride == 0)
      {
        print_protein();
        for (unsigned i=0; i<nprobes;i++)
        {
         // Get coordinates of the reference atom (change to target at some point?)
         int j = n_atoms + i;
         double ref_x = getPosition(j)[0];
         double ref_y = getPosition(j)[1];
         double ref_z = getPosition(j)[2];
         probes[i].print_probe_xyz(i, step);
         probes[i].print_probe_movement(i, step, atoms, n_atoms, ref_x, ref_y, ref_z);
        }
      }

      auto end_psi = high_resolution_clock::now();
      int exec_time = duration_cast<microseconds>(end_psi - start_psi).count();
      // cout << "Step " << step << ": executed in " << exec_time << " microseconds." << endl;

      // if (step>=10) exit(0);

    } // close calculate
  }   // close colvar
} // close plmd
