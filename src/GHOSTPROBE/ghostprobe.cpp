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
#include "aidefunctions.h"

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
      unsigned pertstride;
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
      bool dumpderivatives;
      double err_tol=0.00000000001;
      vector<bool> dxnonull;
      unsigned dxnonull_total;
      vector<double> tx;
      vector<double> ty;
      vector<double> tz;
      double sum_dx;
      double sum_dy;
      double sum_dz;
      double sum_tx;
      double sum_ty;
      double sum_tz;
      arma::vec L;




    public:
      explicit Ghostprobe(const ActionOptions &);
      // active methods:
      void calculate() override;
      void reset();
      void correct_derivatives();
      void print_protein();
      void get_init_crd();
      static void registerKeywords(Keywords &keys);
    };

    PLUMED_REGISTER_ACTION(Ghostprobe, "GHOSTPROBE")

    void Ghostprobe::registerKeywords(Keywords &keys)
    {
      Colvar::registerKeywords(keys);
      keys.addFlag("DEBUG", false, "Running in debug mode");
      keys.addFlag("NOCVCALC", false, "skip CV calculation");
      keys.addFlag("NOUPDATE", false, "skip probe update");
      keys.addFlag("NODXFIX", false, "skip derivative correction");
      keys.addFlag("PERFORMANCE", false, "measure execution time");
      keys.addFlag("DUMPDERIVATIVES", false, "print derivatives and corrections");
      keys.add("atoms", "ATOMS", "Atoms to include in druggability calculations (start at 1)");
      keys.add("atoms", "ATOMS_INIT", "Atoms in which the probes will be initially centered.");
      keys.add("optional", "NPROBES", "Number of probes to use");
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
                                                performance(false),
                                                dumpderivatives(false)
    {
/*
Initialising openMP threads.
This does not seem to be affected by the environment variable $PLUMED_NUM_THREADS
*/    #pragma omp parallel 
      {
      nthreads = omp_get_num_threads();
      ndev = omp_get_num_devices();
      }
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
      parseFlag("DUMPDERIVATIVES",dumpderivatives);

      parseAtomList("ATOMS", atoms);
      n_atoms = atoms.size();

      parseAtomList("ATOMS_INIT", atoms_init);
      n_init = atoms_init.size();

      parse("NPROBES", nprobes);
      if (!nprobes)
      {
        nprobes = 16;
      }
      
      /*
      The following bit checks if ATOMS_INIT has been specified.
      If so, all probes will be initialised in the centre of ATOMS_INIT.
      Otherwise, each probe will be initialised on a protein atom chosen at random.
      */      
      if (n_init==0)
      {
        cout << "geting random atoms_init" << endl;
        for (unsigned i=0; i<nprobes; i++)
        {
          unsigned j_rand=aidefunctions::get_random_integer(0, n_atoms-1);
          init_j.push_back(j_rand);
          cout << j_rand << endl;
        }
        cout << "got random atoms_init" << endl;
      }
      else
      {
        for (unsigned j=0;j<n_init; j++)
        {
          int j_idx=aidefunctions::findIndex(atoms,atoms_init[j]);
          if (j_idx==-1)
          {
            cout << "Atom " << atoms_init[j].index() << " not found in ATOMS. Adding it." << endl;
            atoms.push_back(atoms_init[j]);
            init_j.push_back(atoms.size()-1);
          }
          else
          {
            init_j.push_back(j_idx);
          }
        }
      }

      cout << "Requesting " << atoms.size() << " atoms" << endl;
      requestAtoms(atoms);
      cout << "--------- Initialising Ghostprobe Collective Variable -----------" << endl;

      cout << "Using " << nprobes << " spherical probe(s) with the following parameters:" << endl;

      parse("RMIN", Rmin);
      if (!Rmin)
        Rmin = 0.;
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
        Cmin = 0.; // obtained from generating 10000 random points in VHL's crystal structure
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

      parse("PERTSTRIDE",pertstride);
      if(!pertstride)
      {
        pertstride=1;
      }
      parse("KPERT",kpert);
      if (!kpert)
      {
        kpert=0.001;
      }
      cout << "Perturbations of " << kpert << " nm will be applied to all probes." << endl;
      
      for (unsigned i = 0; i < nprobes; i++)
      {
        probes.push_back(Probe(i,
                               Rmin, deltaRmin, 
                               Rmax, deltaRmax, 
                               Cmin, deltaC, 
                               Pmin, deltaP, 
                               n_atoms, kpert));
        cout << "Probe " << i << " initialised" << endl;
      }

      // parameters used to control output
      parse("PROBESTRIDE", probestride);
      if (!probestride)
        probestride = 1;
      cout << "Information to post-process probe coordinates will be printed every " << probestride << " steps" << endl
           << endl;

      checkRead();

      // Allocate space for atom coordinates

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
        dxnonull_total=0;
        dxnonull=vector<bool>(n_atoms,false);
        tx=vector<double>(n_atoms,0);
        ty=vector<double>(n_atoms,0);
        tz=vector<double>(n_atoms,0);
        L=arma::vec(6);
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

      fill(dxnonull.begin(),dxnonull.end(),false);
      dxnonull_total=0;
      fill(tx.begin(), tx.end(), 0);
      fill(ty.begin(), ty.end(), 0);
      fill(tz.begin(), tz.end(), 0);
      sum_dx = 0;
      sum_dy = 0;
      sum_dz = 0;
      sum_tx = 0;
      sum_ty = 0;
      sum_tz = 0;
    }

    void Ghostprobe::correct_derivatives()
    {
      //cout << "entering derivatives correction" << endl;
      if (step==0 and dumpderivatives)
      {
        ofstream wfile;
        wfile.open("derivatives.csv");
        wfile << "Step Atom " << 
                 "dx correction_dx corrected_dx " << 
                 "dy correction_dy corrected_dy " << 
                 "dz correction_dz corrected_dz" << endl;
        wfile.close();
      }

      // auto point0=high_resolution_clock::now();

      //cout << "Step 1: calculate sums of derivatives and sums of torques in each direction" << endl;
      for (unsigned j = 0; j < n_atoms; j++)
      {
        if (d_Psi_dx[j]==0 and d_Psi_dy[j]==0 and d_Psi_dz[j]==0)
            continue;
        dxnonull[j]=true;
        dxnonull_total++;

        tx[j]=atoms_y[j] * d_Psi_dz[j] - atoms_z[j] * d_Psi_dy[j];
        ty[j]=atoms_z[j] * d_Psi_dx[j] - atoms_x[j] * d_Psi_dz[j];
        tz[j]=atoms_x[j] * d_Psi_dy[j] - atoms_y[j] * d_Psi_dx[j];

        sum_dx += d_Psi_dx[j];  
        sum_dy += d_Psi_dy[j];
        sum_dz += d_Psi_dz[j];
        sum_tx += tx[j];
        sum_ty += ty[j];
        sum_tz += tz[j];
      }

      //cout << "Step 2: Build matrix A" << endl;
      unsigned k=0;
      arma::mat A(6,dxnonull_total*3);
      for (unsigned j=0; j<n_atoms;j++)
      {
        if (!dxnonull[j])
           continue;

        A.row(0).col(k+0*dxnonull_total) = 1.0;
        A.row(1).col(k+0*dxnonull_total) = 0.0;
        A.row(2).col(k+ 0*dxnonull_total) = 0.0;
        A.row(3).col(k+ 0*dxnonull_total) = 0.0;
        A.row(4).col(k+ 0*dxnonull_total) = atoms_z[j];
        A.row(5).col(k+ 0*dxnonull_total) = -atoms_y[j];
        
        A.row(0).col(k+ 1*dxnonull_total) = 0.0;
        A.row(1).col(k+ 1*dxnonull_total) = 1.0;
        A.row(2).col(k+ 1*dxnonull_total) = 0.0;
        A.row(3).col(k+ 1*dxnonull_total) = -atoms_z[j];
        A.row(4).col(k+ 1*dxnonull_total) = 0.0;
        A.row(5).col(k+ 1*dxnonull_total) = atoms_x[j];
        
        A.row(0).col(k+ 2*dxnonull_total) = 0.0;
        A.row(1).col(k+ 2*dxnonull_total) = 0.0;
        A.row(2).col(k+ 2*dxnonull_total) = 1.0;
        A.row(3).col(k+ 2*dxnonull_total) = atoms_y[j];
        A.row(4).col(k+ 2*dxnonull_total) = -atoms_x[j];
        A.row(5).col(k+ 2*dxnonull_total) = 0.0;
  
        k++;
      }

      //cout << "Step 3: Build matrix L" << endl;
      L[0] = -sum_dx;
      L[1] = -sum_dy;
      L[2] = -sum_dz;
      L[3] = -sum_tx;
      L[4] = -sum_ty;
      L[5] = -sum_tz;

      //cout << "Step 4: calculate constants" << endl;
      arma::mat At = trans(A);
      arma::vec c = At*pinv(A*At)*L;
      //c.print();

      //cout << "Step 5 Apply constants" << endl;
      k=0;
      for (unsigned j = 0; j < n_atoms; j++)
      {
        if (!dxnonull[j])
           continue;
        if (dumpderivatives and step%probestride==0)
        {
         ofstream wfile;
         wfile.open("derivatives.csv",std::ios_base::app);
         wfile << setprecision(16); //need to save all significant figures for python postprocessing!
         wfile << step << " " << j << " " << d_Psi_dx[j] << " " << c[k + 0 * dxnonull_total] << " " << d_Psi_dx[j]+c[k + 0 * dxnonull_total] << 
                                      " " << d_Psi_dy[j] << " " << c[k + 1 * dxnonull_total] << " " << d_Psi_dy[j]+c[k + 1 * dxnonull_total] << 
                                      " " << d_Psi_dz[j] << " " << c[k + 2 * dxnonull_total] << " " << d_Psi_dz[j]+c[k + 2 * dxnonull_total] << endl;
         wfile.close(); 
        }
        d_Psi_dx[j] += c[k + 0 * dxnonull_total];
        d_Psi_dy[j] += c[k + 1 * dxnonull_total];
        d_Psi_dz[j] += c[k + 2 * dxnonull_total];
        k++;
      } 
      
      //cout << "Step 6: Sanity check" << endl;
      sum_dx=0;
      sum_dy=0;
      sum_dz=0;
      sum_tx=0;
      sum_ty=0;
      sum_tz=0;

      for (unsigned j=0;j<n_atoms;j++)
      {
       sum_dx+=d_Psi_dx[j];
       sum_dy+=d_Psi_dy[j];
       sum_dz+=d_Psi_dz[j];
       sum_tx+=atoms_y[j]*d_Psi_dz[j]-atoms_z[j]*d_Psi_dy[j];
       sum_ty+=atoms_z[j]*d_Psi_dx[j]-atoms_x[j]*d_Psi_dz[j];
       sum_tz+=atoms_x[j]*d_Psi_dy[j]-atoms_y[j]*d_Psi_dx[j];
      }

      if ((sum_dx>err_tol) or (sum_dy>err_tol) or (sum_dz>err_tol) or 
          (sum_tx>err_tol) or (sum_ty>err_tol) or (sum_tz>err_tol))
      {
      cout << "Error: Correction of derivatives malfunctioned. Simulation will now end." << endl;
      cout << "Sum derivatives: " << sum_dx << " " << sum_dy << " " << sum_dz << endl;
      cout << "Sum torques: "     << sum_tx << " " << sum_ty << " " << sum_tz << endl;
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
     //cout << "exiting derivatives correction" << endl;
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

    void Ghostprobe::get_init_crd()
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
          probes[i].perturb_probe();
          cout << "Probe " << i << " centered on atom " << atoms[init_j[i]].serial() << endl;
        }
      }
      else
      {
       //cout << n_atoms << endl;
       for (unsigned j=0; j<init_j.size();j++)
       {
        //cout << j << " " << init_j[j] << " " << getPosition(init_j[j])[0]<< " " << getPosition(init_j[j])[1]<< " " << getPosition(init_j[j])[2] <<  endl;
        x+=getPosition(init_j[j])[0]/init_j.size();
        y+=getPosition(init_j[j])[1]/init_j.size();
        z+=getPosition(init_j[j])[2]/init_j.size();
       }
       for (unsigned i = 0; i < nprobes; i++)
       {
        probes[i].place_probe(x,y,z);
       }
       //cout << "All probes are initialised at point " << x << " " << y << " " << z << endl;
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
      
      // Get atom positions
      for (unsigned j = 0; j < n_atoms; j++)
      {
        atoms_x[j] = getPosition(j)[0];
        atoms_y[j] = getPosition(j)[1];
        atoms_z[j] = getPosition(j)[2];
      }

      // At step 0, place the probes using get_init_crd()
      if (step==0 or noupdate)
         get_init_crd();

      #pragma omp parallel for 
      for (unsigned i = 0; i < nprobes; i++)
      {
        // Update probe coordinates
        if (!noupdate)
        {
         probes[i].move_probe(step, atoms_x, atoms_y, atoms_z);
        }
        //perturb probe coordinates  at every step
        probes[i].perturb_probe();

        //Calculate Psi and its derivatives
        if (!nocvcalc)
        {
          probes[i].calculate_activity(atoms_x, atoms_y, atoms_z);
          #pragma omp critical //avoid race condition
          {
           Psi+=probes[i].activity/nprobes;
           for (unsigned j=0;j<n_atoms;j++)
           {
             d_Psi_dx[j]+=probes[i].d_activity_dx[j]/nprobes;
             d_Psi_dy[j]+=probes[i].d_activity_dy[j]/nprobes;
             d_Psi_dz[j]+=probes[i].d_activity_dz[j]/nprobes;
           }
          }
        }

        if (step%probestride==0)
        {
          probes[i].print_probe_xyz(step);
          probes[i].print_probe_movement(step, atoms, n_atoms);
        }
      }

      //Correct the Psi derivatives so that they sum 0
      if (!nodxfix)
      {
       correct_derivatives();
      }
   
      //Send Psi and derivatives back to Plumed
      setValue(Psi);
      for (unsigned j=0;j<n_atoms;j++)
      {
        setAtomsDerivatives(j,Vector(d_Psi_dx[j],d_Psi_dy[j],d_Psi_dz[j]));
      }

      //print output for post_processing
       if (step % probestride == 0)
       {
         print_protein();
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
