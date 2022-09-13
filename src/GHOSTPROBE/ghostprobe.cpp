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

    class Ghostprobe : public Colvar
    {
      // Execution control variables
      int nthreads=0;     // number of available OMP threads
      int ndev=0;         // number of available OMP accelerators
      bool performance; // print execution time
      // MD control variables
      bool pbc;
      // CV control variables
      bool nocvcalc;
      bool nodxfix;
      bool noupdate;
      double kpert=0;
      unsigned pertstride=0;

      // Parameters
      double Rmin=0;          // mind below which an atom is considered to be clashing with the probe
      double deltaRmin=0;        // interval over which contact terms are turned on and off
      double Rmax=0;          // distance above which an atom is considered to be too far away from the probe*
      double deltaRmax=0;        // interval over which contact terms are turned on and off
      double Pmin=0;           // packing factor below which depth term equals 0
      double deltaP=0;         // interval over which depth term turns from 0 to 1
      double Cmin=0;           // packing factor below which depth term equals 0
      double deltaC=0;         // interval over which depth term turns from 0 to 1

      // Set up of CV
      vector<PLMD::AtomNumber> atoms; // indices of atoms supplied to the CV (starts at 1)
      unsigned n_atoms=0;               // number of atoms supplied to the CV
      vector<double> atoms_x;
      vector<double> atoms_y;
      vector<double> atoms_z;
      unsigned step=0;

      vector<PLMD::AtomNumber> atoms_init; // Indices of the atoms in which the probes will be initially centered
      unsigned n_init=0;                     // number of atoms used in ATOMS_INIT
      vector<unsigned> init_j;             // Indices of atoms_init in getPositions()

      vector<Probe> probes; // This will contain all the spherical probes
      unsigned nprobes=0;     // number of spherical probes to use

      // Output control variables
      unsigned probestride=0; // stride to print information for post-processing the probe coordinates

      // Calculation of CV and its derivatives

      double Psi=0;
      vector<double> d_Psi_dx;
      vector<double> d_Psi_dy;
      vector<double> d_Psi_dz;

      // Correction of derivatives
      double sum_d_dx;
      double sum_d_dy;
      double sum_d_dz;
      double sum_t_dx;
      double sum_t_dy;
      double sum_t_dz;

      arma::vec L;
      unsigned nrows=0;
      unsigned ncols=0;
      arma::mat A;
      arma::mat Aplus;
      arma::vec P;
      vector<double> sum_P;
      vector<double> sum_rcrossP;
      //for when correction of derivatives fails
      double err_tol=0.00000001; //1e-8

    public:
      explicit Ghostprobe(const ActionOptions &);
      // active methods:
      void calculate() override;
      void reset();
      void correct_derivatives();
      void print_protein();
      static void registerKeywords(Keywords &keys);
      void get_init_crd(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z);
    };

    PLUMED_REGISTER_ACTION(Ghostprobe, "GHOSTPROBE")

    void Ghostprobe::registerKeywords(Keywords &keys)
    {
      Colvar::registerKeywords(keys);
      keys.addFlag("DEBUG", false, "Running in debug mode");
      keys.addFlag("NOCVCALC", false, "skip CV calculation");
      keys.addFlag("NOUPDATE", false, "skip probe update");
      keys.addFlag("NODXFIX", false, "skip derivative correction");
      keys.addFlag("TABOO", false, "skip derivative correction");
      keys.addFlag("PERFORMANCE", false, "measure execution time");
      keys.add("atoms", "ATOMS", "Atoms to include in druggability calculations (start at 1)");
      keys.add("atoms", "ATOMS_INIT", "Atoms in which the probes will be initially centered.");
      keys.add("optional", "NPROBES", "Number of probes to use");
      keys.add("optional", "RPROBE", "Radius of every probe in nm");
      keys.add("optional", "PROBESTRIDE", "Print probe coordinates info every PROBESTRIDE steps");
      keys.add("optional", "RMIN", "");
      keys.add("optional", "DELTARMIN", "");
      keys.add("optional", "RMAX", "");
      keys.add("optional", "DELTARMAX", "");
      keys.add("optional", "CMIN", "");
      keys.add("optional", "DELTAC", "");
      keys.add("optional", "PMIN", "");
      keys.add("optional", "DELTAP", "");
      keys.add("optional", "KPERT", "");
      keys.add("optional", "PERTSTRIDE", "");
    }

    Ghostprobe::Ghostprobe(const ActionOptions &ao) : PLUMED_COLVAR_INIT(ao),
                                                pbc(true),
                                                nocvcalc(false),
                                                noupdate(false),
                                                nodxfix(false),
                                                performance(false)
    {
/*
Initialising openMP threads.
This does not seem to be affected by the environment variable $PLUMED_NUM_THREADS
*/
#pragma omp parallel
      nthreads = omp_get_num_threads();
      ndev = omp_get_num_devices();
      cout << "------------ Available Computing Resources -------------" << endl;
      cout << "Ghostprobe initialised with " << nthreads << " OMP threads " << endl;
      cout << "and " << ndev << " OMP compatible accelerators (not currently used)" << endl;

      addValueWithDerivatives();
      setNotPeriodic();

      bool nopbc = !pbc;
      parseFlag("NOPBC", nopbc);
      pbc = !nopbc;

      parseFlag("NOCVCALC",nocvcalc);
      parseFlag("NOUPDATE", noupdate);
      parseFlag("NODXFIX", nodxfix);
      parseFlag("PERFORMANCE", performance);

      parseAtomList("ATOMS", atoms);
      parseAtomList("ATOMS_INIT", atoms_init);

      n_atoms = atoms.size();
      n_init = atoms_init.size();

      parse("NPROBES", nprobes);
      if (!nprobes)
      {
        nprobes = 16;
      }
      //The following bit checks if ATOMS_INIT are already in ATOMS.
      //Those that are are not requested twice.
      if (atoms_init.size()>0)
      {
        for (unsigned k = 0; k < n_init; k++)
        {
          for (unsigned j=0; j<n_atoms; j++)
          {
            if (atoms_init[k]==atoms[j])
            {
             init_j.push_back(j);
             break; 
            }
            if (j==(n_atoms-1)) // This means we reached the last atom in ATOMS
            {
              cout << "Atom " << atoms_init[k].serial() << " not found in ATOMS. Adding it." << endl;
              atoms.push_back(atoms_init[k]);
              init_j.push_back(atoms.size()-1);
            }
          }
        }
      }
      else
      {
        for (unsigned i=0; i<nprobes; i++)
        {
        random_device rd;                                   // only used once to initialise (seed) engine
        mt19937 rng(rd());                                  // random-number engine used (Mersenne-Twister in this case)
        uniform_int_distribution<unsigned> uni(0, n_atoms); // guaranteed unbiased
        auto random_integer = uni(rng);
        init_j.push_back(random_integer);
        }
      }

      cout << "Requesting " << atoms.size() << " atoms" << endl;
      requestAtoms(atoms);
      cout << "--------- Initialising Ghostprobe Collective Variable -----------" << endl;

      cout << "Using " << nprobes << " spherical probe(s) with the following parameters:" << endl;

      parse("RMIN", Rmin);
      if (!Rmin)
        Rmin = 0.25;
      cout << "Rmin = " << Rmin << " nm" << endl;

      parse("DELTARMIN", deltaRmin);
      if (!deltaRmin)
        deltaRmin = 0.15;
      cout << "deltaRmin = " << deltaRmin << " nm" << endl;

      parse("RMAX", Rmax);
      if (!Rmax)
        Rmax = 0.6;
      cout << "Rmax = " << Rmax << " nm" << endl;

      parse("DELTARMAX", deltaRmax);
      if (!deltaRmax)
        deltaRmax = 0.15;
      cout << "deltaRmax = " << deltaRmax << " nm" << endl;

      parse("CMIN", Cmin);
      if (!Cmin)
        Cmin = 0; // obtained from generating 10000 random points in VHL's crystal structure
      cout << "CMIN = " << Cmin << endl;

      parse("DELTAC", deltaC);
      if (!deltaC)
        deltaC = 7; // obtained from generating 10000 random points in VHL's crystal structure
      cout << "DELTAC = " << deltaC << endl;

      parse("PMIN", Pmin);
      if (!Pmin)
        Pmin = 25; // obtained from generating 10000 random points in VHL's crystal structure
      cout << "PMIN = " << Pmin << endl;

      parse("DELTAP", deltaP);
      if (!deltaP)
        deltaP = 15; // obtained from generating 10000 random points in VHL's crystal structure
      cout << "DELTAP = " << deltaP << endl;

      parse("KPERT",kpert);
      parse("PERTSTRIDE",pertstride);
      if (!kpert or !pertstride)
      {
        kpert=0.001;
        pertstride=-1;
        cout << " KPERT and/or PERTRSTRIDE not set -- POCKET SEARCH WILL NOT BE DONE " << endl;
      }
      else
      {
        cout << "Probe will be perturbed every " << pertstride << " steps.";
        cout << "The perturbation will be of " << kpert << " nm." << endl;
      }
      
      for (unsigned i = 0; i < nprobes; i++)
      {
        probes.push_back(Probe(i,
                               Rmin, deltaRmin, 
                               Rmax, deltaRmax, 
                               Cmin, deltaC, 
                               Pmin, deltaP, 
                               n_atoms, kpert,
                               init_j[i]));
        cout << "Probe " << i << " initialised" << endl;
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

      cout << "---------Initialisng Ghostprobe and its derivatives---------" << endl;
      Psi = 0;
      d_Psi_dx = vector<double>(n_atoms, 0);
      d_Psi_dy = vector<double>(n_atoms, 0);
      d_Psi_dz = vector<double>(n_atoms, 0);

      if (nocvcalc)
         cout << "WARNING: NOCVCALC flag has been included. CV will NOT be calculated." << endl;

      if (!nodxfix)
      {
        cout << "---------Initialisng correction of Ghostprobe derivatives---------" << endl;

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
        cout << "Ghostprobe derivatives are not going to be corrected" << endl;
        cout << "Use the NODXFIX flag with care, as this means that" << endl;
        cout << "the sum of forces in the system will not be zero" << endl;
      }

      cout << "--------- Initialisation complete -----------" << endl;
    }

    // reset Ghostprobe and derivatives to 0
    void Ghostprobe::reset()
    {
      Psi = 0;
      fill(d_Psi_dx.begin(), d_Psi_dx.end(), 0);
      fill(d_Psi_dy.begin(), d_Psi_dy.end(), 0);
      fill(d_Psi_dz.begin(), d_Psi_dz.end(), 0);

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

    void Ghostprobe::correct_derivatives()
    {
      
      // auto point0=high_resolution_clock::now();
      // step 0: calculate sums of derivatives and sums of torques in each direction
      for (unsigned j = 0; j < n_atoms; j++)
      {
        sum_d_dx += d_Psi_dx[j];
        sum_d_dy += d_Psi_dy[j];
        sum_d_dz += d_Psi_dz[j];

        sum_t_dx += atoms_y[j] * d_Psi_dz[j] - atoms_z[j] * d_Psi_dy[j];
        sum_t_dy += atoms_z[j] * d_Psi_dx[j] - atoms_x[j] * d_Psi_dz[j];
        sum_t_dz += atoms_x[j] * d_Psi_dy[j] - atoms_y[j] * d_Psi_dx[j];
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
          A.row(3).col(j) = -atoms_z[j - n_atoms];
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
        d_Psi_dx[j] += P[j + 0 * n_atoms];
        d_Psi_dy[j] += P[j + 1 * n_atoms];
        d_Psi_dz[j] += P[j + 2 * n_atoms];
      }
      // auto point5=high_resolution_clock::now();

      // Only for debugging
      
      sum_d_dx=0;
      sum_d_dy=0;
      sum_d_dz=0;
      sum_t_dx=0;
      sum_t_dy=0;
      sum_t_dz=0;

      for (unsigned j=0;j<n_atoms;j++)
      {
       sum_d_dx+=d_Psi_dx[j];
       sum_d_dy+=d_Psi_dy[j];
       sum_d_dz+=d_Psi_dz[j];

       sum_t_dx+=atoms_y[j]*d_Psi_dz[j]-atoms_z[j]*d_Psi_dy[j];
       sum_t_dy+=atoms_z[j]*d_Psi_dx[j]-atoms_x[j]*d_Psi_dz[j];
       sum_t_dz+=atoms_x[j]*d_Psi_dy[j]-atoms_y[j]*d_Psi_dx[j];
      }

      if ((sum_d_dx>err_tol) or (sum_d_dy>err_tol) or (sum_d_dz>err_tol) or 
          (sum_t_dx>err_tol) or (sum_t_dy>err_tol) or (sum_t_dz>err_tol))
      {
      cout << "Error: Correction of derivatives malfunctioned. Simulation will now end." << endl;
      cout << "Sum derivatives: " << sum_d_dx << " " << sum_d_dy << " " << sum_d_dz << endl;
      cout << "Sum torques: " << sum_t_dx << " " << sum_t_dy << " " << sum_t_dz << endl;
      exit(0);
      }
      
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

    void Ghostprobe::print_protein()
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

    void Ghostprobe::get_init_crd(vector<double> atoms_x, vector<double> atoms_y, vector<double> atoms_z)
    {
      double x=0;
      double y=0;
      double z=0;

      if (atoms_init.size() == 0)
      {
        for (unsigned i = 0; i < nprobes; i++)
        {
          x=getPosition(init_j[i])[0];
          y=getPosition(init_j[i])[1];
          z=getPosition(init_j[i])[2];
          probes[i].place_probe(x,y,z);
          probes[i].perturb_probe(0,atoms_x,atoms_y,atoms_z);
          cout << "Probe " << i << "Centered on atom " << atoms[init_j[i]].serial();
        }
      }
      else
      {
       cout << n_atoms << endl;
       for (unsigned j=0; j<init_j.size();j++)
       {
        cout << j << " " << init_j[j] << " " << getPosition(init_j[j])[0]<< " " << getPosition(init_j[j])[1]<< " " << getPosition(init_j[j])[2] <<  endl;
        x+=getPosition(init_j[j])[0]/init_j.size();
        y+=getPosition(init_j[j])[1]/init_j.size();
        z+=getPosition(init_j[j])[2]/init_j.size();
       }
       for (unsigned i = 0; i < nprobes; i++)
       {
        probes[i].place_probe(x,y,z);
       }
       cout << "All probes are initialised at point " << x << " " << y << " " << z << endl;
      }
    }

    // calculator
    void Ghostprobe::calculate()
    {
      //arma::arma_version ver;
      //cout << ver.as_string() << endl;
      //exit(0);
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
      }

      if (step==0)
         get_init_crd(atoms_x,atoms_y,atoms_z);

      #pragma omp parallel for
      for (unsigned i = 0; i < nprobes; i++)
      {
        // Update probe coordinates
        if (!noupdate)
        {
         probes[i].move_probe(step, atoms_x, atoms_y, atoms_z);
        }

        if (!nocvcalc)
        {
          probes[i].calculate_activity(atoms_x, atoms_y, atoms_z);
          Psi+=probes[i].activity/nprobes;
          for (unsigned j=0;j<n_atoms;j++)
          {
            d_Psi_dx[j]+=probes[i].d_activity_dx[j]/nprobes;
            d_Psi_dy[j]+=probes[i].d_activity_dy[j]/nprobes;
            d_Psi_dz[j]+=probes[i].d_activity_dz[j]/nprobes;
          }
        }
        
        if (step%pertstride==0 and step>0 and (!nodxfix))
           probes[i].perturb_probe(step,atoms_x,atoms_y,atoms_z);
      }

      if (!nodxfix)
      {
       correct_derivatives();
      }
   
      setValue(Psi);
      for (unsigned j=0;j<n_atoms;j++)
      {
        setAtomsDerivatives(j,Vector(d_Psi_dx[j],d_Psi_dy[j],d_Psi_dz[j]));
      }

      //print output for post_processing
       if (step % probestride == 0)
       {
         print_protein();
         for (unsigned i=0; i<nprobes;i++)
         {
          // Get coordinates of the reference atom
          unsigned j = init_j[i];

          double ref_x = getPosition(j)[0];
          double ref_y = getPosition(j)[1];
          double ref_z = getPosition(j)[2];
          probes[i].print_probe_xyz(i, step);
          probes[i].print_probe_movement(i, step, atoms, n_atoms, ref_x, ref_y, ref_z);
         }
       }
      
      if (performance)
      {
       auto end_psi = high_resolution_clock::now();
       int exec_time = duration_cast<microseconds>(end_psi - start_psi).count();
       cout << "Step " << step << ": executed in " << exec_time << " microseconds." << endl;
      }

      // if (step>=10) exit(0);

    } // close calculate
  }   // close colvar
} // close plmd
