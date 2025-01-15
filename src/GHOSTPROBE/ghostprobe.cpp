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
#include "core/ActionRegister.h"

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
      //calculation speed
      bool performance;
      time_point<high_resolution_clock> start_psi;
      time_point<high_resolution_clock> end_psi;
      time_point<high_resolution_clock> start_dxfix;
      time_point<high_resolution_clock> end_dxfix;

      // All of these are just for correct_derivatives()
      time_point<high_resolution_clock> start_tor;
      time_point<high_resolution_clock> end_tor;
      time_point<high_resolution_clock> start_A;
      time_point<high_resolution_clock> end_A;
      time_point<high_resolution_clock> start_B;
      time_point<high_resolution_clock> end_B;
      time_point<high_resolution_clock> start_Bt;
      time_point<high_resolution_clock> end_Bt;
      time_point<high_resolution_clock> start_c;
      time_point<high_resolution_clock> end_c;
      time_point<high_resolution_clock> start_correction;
      time_point<high_resolution_clock> end_correction;
      time_point<high_resolution_clock> start_test;
      time_point<high_resolution_clock> end_test;

      // MD control variables
      bool pbc;
      unsigned step=0; // initial step (so that it doesn't get initialised at a random unsigned)

      // CV control variables
      bool nocvcalc; // DO NOT CALCULATE THE CV (for testing of probe update)
      bool nodxfix; // DO NOT CORRECT THE DERIVATIVES (for testing that the derivatives are proper implemented)
      bool noupdate; // DO NOT UPDATE PROBE COORDINATES (to test the implementation of the CV)
      unsigned pertstride=0; // stride for the perturbation of the probe position
      
      // GHOSTPROBE Parameters
      double Rmin=0;          // mind below which an atom is considered to be clashing with the probe
      double deltaRmin=0;        // interval over which contact terms are turned on and off
      double Rmax=0;          // distance above which an atom is considered to be too far away from the probe*
      double deltaRmax=0;        // interval over which contact terms are turned on and off
      double Pmin=0;           // packing factor below which depth term equals 0
      double deltaP=0;         // interval over which depth term turns from 0 to 1
      double Cmin=0;           // packing factor below which depth term equals 0
      double deltaC=0;         // interval over which depth term turns from 0 to 1
      double kpert=0; // Perturbation of the probe position when C>0 and P>0
      double kxplor=0; // Perturbation of the probe position when C=0 and P>0

      // Variables related to atoms
      vector<PLMD::AtomNumber> atoms; // indices of atoms supplied to the CV (starts at 1)
      unsigned n_atoms=0;               // number of atoms supplied to the CV
      vector<double> atoms_x;
      vector<double> atoms_y;
      vector<double> atoms_z;
      bool protein_out;
      
      // Variables required to initialise the probe
      Probe probe; // the probe object
      string probe_label; // label to distinguish the probes
      string probe_init_pdb; // pdb with original probe position (optional)
      PDB probe_init; // PDB object for probe_init_pdb
      double Psi=0;
      vector<double> d_Psi_dx;
      vector<double> d_Psi_dy;
      vector<double> d_Psi_dz;

      // Output control variables
      unsigned probestride=0; // stride to print information for post-processing the probe coordinates

      // Correction of derivatives
      vector<double> tx;
      vector<double> ty;
      vector<double> tz;
      vector<bool> dxnonull;
      unsigned dxnonull_size;
      double sum_d_dx;
      double sum_d_dy;
      double sum_d_dz;
      double sum_t_dx;
      double sum_t_dy;
      double sum_t_dz;

      //for when correction of derivatives fails
      arma::mat A;
      arma::mat B;
      arma::mat Bt;
      arma::mat BtB;
      arma::vec v;
      arma::vec Btv;
      arma::vec x;
      arma::vec c;

      double err_tol=1e-8;
      bool dumpderivatives;

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
      keys.addFlag("PRINT_PROTEIN", false, "Print protein structure");
      keys.add("atoms", "ATOMS", "Atoms to include in druggability calculations (start at 1)");
      keys.add("optional", "RMIN", "");
      keys.add("optional", "DELTARMIN", "");
      keys.add("optional", "RMAX", "");
      keys.add("optional", "DELTARMAX", "");
      keys.add("optional", "CMIN", "");
      keys.add("optional", "DELTAC", "");
      keys.add("optional", "PMIN", "");
      keys.add("optional", "DELTAP", "");
      keys.add("optional", "KPERT", "");
      keys.add("optional", "KXPLOR", "");
      keys.add("optional", "PROBESTRIDE", "Print probe coordinates info every PROBESTRIDE steps");
      keys.add("optional", "PERTSTRIDE", "Do a full KPERT random perturbation every PERTSTRIDE steps");
      keys.add("optional","PROBE_INIT","Coordinates od reference ligand atoms to place the probes on.");
      keys.add("optional","PROBE_LABEL","LABEL string of the present probe");
    }

    Ghostprobe::Ghostprobe(const ActionOptions &ao) : PLUMED_COLVAR_INIT(ao),
                                                pbc(true),
                                                nocvcalc(false),
                                                noupdate(false),
                                                nodxfix(false),
                                                performance(false),
                                                dumpderivatives(false),
                                                protein_out(false)
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

      addValueWithDerivatives();
      setNotPeriodic();

      bool nopbc = !pbc;
      parseFlag("NOPBC", nopbc);
      pbc = !nopbc;

      parseFlag("NOCVCALC",nocvcalc);
      parseFlag("NOUPDATE", noupdate);
      parseFlag("NODXFIX", nodxfix);
      parseFlag("PERFORMANCE", performance);
      if (performance)
      {
       ofstream wfile;
       wfile.open("performance.txt");
       wfile << "Psi correction" << endl;
       wfile.close();
       
       //ofstream wfile;
       wfile.open("performance_dxfix.txt");
       wfile << "torques A B Bcoot Bt c correction test total" << endl;
       wfile.close();
      }

      parseFlag("DUMPDERIVATIVES",dumpderivatives);
      if (dumpderivatives)
      {
        ofstream wfile;
        wfile.open("derivatives.csv");
        wfile << "Step Atom dx dy dz tx ty tz correction" << endl;
        wfile.close();
      }

      parseAtomList("ATOMS", atoms);
      n_atoms = atoms.size();
      

      parse("PROBE_INIT",probe_init_pdb);
      
      if (!probe_init_pdb.empty()) 
      {
        if( !probe_init.read(probe_init_pdb,usingNaturalUnits(),0.1/getUnits().getLength()) )
         {
          cout << "missing input file " << probe_init_pdb << endl;
          exit(1);
         }
      }

      parse("PROBE_LABEL",probe_label);
      if(probe_label=="")
      {
        unsigned randint=aidefunctions::get_random_integer(0,99999);
        probe_label=to_string(randint);
      }

      parseFlag("PRINT_PROTEIN",protein_out);

      
      cout << "Requesting " << atoms.size() << " atoms" << endl;
      requestAtoms(atoms);
      cout << "--------- Initialising Ghostprobe Collective Variable -----------" << endl;

      cout << "GHOSTPROBE parameter set:" << endl;

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
        Cmin = 0.; 
      cout << "CMIN = " << Cmin << endl;

      parse("DELTAC", deltaC);
      if (!deltaC)
        deltaC = 5; 
      cout << "DELTAC = " << deltaC << endl;

      parse("PMIN", Pmin);
      if (!Pmin)
        Pmin = 0; 
      cout << "PMIN = " << Pmin << endl;

      parse("DELTAP", deltaP);
      if (!deltaP)
        deltaP = 17; 
      cout << "DELTAP = " << deltaP << endl;

      parse("PERTSTRIDE",pertstride);

      parse("KPERT",kpert);
      if (!kpert)
      {
        cout << "****************************************************************************" << endl;
        cout << "KPERT HAS either not been set, or manually set to zero." << endl;
        cout << "WARNING: PROBE WILL NOT BE PERTURBED AND POCKET SEARCH WILL NOT BE PERFORMED" << endl;
        cout << "****************************************************************************" << endl;
      }
      else
      {
      cout << "Perturbations of " << kpert << " nm will be applied to all probes." << endl;
      cout << "Perturbations will go in the direction opposite to the derivatives of the activity" << endl;
      cout << "with respect to the probe, when possible. (Fx=-dV/dx)" << endl;
      }

      parse("KXPLOR",kxplor);
      if (kxplor)
      {
      cout << "Perturbations of " << kxplor << " nm will be applied to probes with C equal to 0." << endl;
      cout << "Those perturbations will go in a random direction." << endl;
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
      probe.Probe_init(probe_label,
                    Rmin, deltaRmin, 
                    Rmax, deltaRmax, 
                    Cmin, deltaC, 
                    Pmin, deltaP, 
                    kpert, kxplor,pertstride,
                    n_atoms);
      cout << "Probe " << probe.probe_id << " initialised" << endl;
      Psi = 0;
      d_Psi_dx = vector<double>(n_atoms, 0);
      d_Psi_dy = vector<double>(n_atoms, 0);
      d_Psi_dz = vector<double>(n_atoms, 0);

      if (nocvcalc)
         cout << "WARNING: NOCVCALC flag has been included. CV will NOT be calculated." << endl;

      if (!nodxfix)
      {
        cout << "---------Initialisng correction of Ghostprobe derivatives---------" << endl;
        tx=vector<double>(n_atoms,0);
        ty=vector<double>(n_atoms,0);
        tz=vector<double>(n_atoms,0);
        dxnonull=vector<bool>(n_atoms,false);
        dxnonull_size=0;
        A=arma::mat(6,n_atoms);
        B=arma::mat(n_atoms,n_atoms);
        Bt=arma::mat(n_atoms,n_atoms);
        BtB=arma::mat(n_atoms,n_atoms);
        Btv=arma::vec(n_atoms);
        x=arma::vec(n_atoms);
        v=arma::vec(n_atoms);
        c=arma::vec(n_atoms);
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
      fill(dxnonull.begin(), dxnonull.end(), false);
      dxnonull_size=0;
      fill(tx.begin(),tx.end(),0);
      fill(ty.begin(),ty.end(),0);
      fill(tz.begin(),tz.end(),0);
      sum_d_dx = 0;
      sum_d_dy = 0;
      sum_d_dz = 0;
      sum_t_dx = 0;
      sum_t_dy = 0;
      sum_t_dz = 0;
    }

    void Ghostprobe::correct_derivatives()
    {
      if (performance and step%probestride==0)  start_dxfix = high_resolution_clock::now();
      //cout << "Step 0: calculating torques" << endl;
      if (performance and step%probestride==0)  start_tor = high_resolution_clock::now();
      for (unsigned j=0; j<n_atoms;j++)
      {
        if (d_Psi_dx[j]==0 and d_Psi_dy[j]==0 and d_Psi_dz[j]==0)
           continue;
        //cout << j << " " << d_Psi_dx[j] << " " << d_Psi_dy[j] << " " << d_Psi_dz[j] << endl;
        tx[j]=atoms_y[j]*d_Psi_dz[j]-atoms_z[j]*d_Psi_dy[j];
        ty[j]=atoms_z[j]*d_Psi_dx[j]-atoms_x[j]*d_Psi_dz[j];
        tz[j]=atoms_x[j]*d_Psi_dy[j]-atoms_y[j]*d_Psi_dx[j];
        dxnonull[j]=true;
        dxnonull_size++;
      }
       //if all derivatives are equal to 0 skip this step
      if (dxnonull_size==0)
      {
        //cout << "All derivatives are zero. Skipping correction of derivatives." << endl;
        return;
      }
      if (performance and step%probestride==0)  end_tor = high_resolution_clock::now();

      //cout << "Generating matrices" << endl;
      if (performance and step%probestride==0)  start_A = high_resolution_clock::now();
      v=arma::vec(dxnonull_size);
      fill(v.begin(),v.end(),1);

      A=arma::mat(6,dxnonull_size);
      unsigned k=0;
      for (unsigned j=0; j<n_atoms; j++)
      {
        if (!dxnonull[j])
            continue;
        A.row(0).col(k)=d_Psi_dx[j];
        A.row(1).col(k)=d_Psi_dy[j];
        A.row(2).col(k)=d_Psi_dz[j];
        A.row(3).col(k)=tx[j];
        A.row(4).col(k)=ty[j];
        A.row(5).col(k)=tz[j];
        k++;
      }
      if (performance and step%probestride==0)  end_A = high_resolution_clock::now();
      
      //cout << "Matrix ops" << endl;
      //Apply https://math.stackexchange.com/questions/4686718/how-to-solve-a-linear-system-with-more-variables-than-equations-with-constraints/4686826#4686826
      if (performance and step%probestride==0)  start_B = high_resolution_clock::now();
      B=arma::null(A);
      if (performance and step%probestride==0)  end_B = high_resolution_clock::now();

      if (performance and step%probestride==0)  start_Bt = high_resolution_clock::now();
      Bt=B.t();
      if (performance and step%probestride==0)  end_Bt = high_resolution_clock::now();

      if (performance and step%probestride==0)  start_c = high_resolution_clock::now();
      c=(B*arma::inv_sympd(Bt*B)*Bt*v); //if matrix isn't invertible, pinv() will provide the best approximation

      if (performance and step%probestride==0)  end_c = high_resolution_clock::now();
      
      //cout << "Assigning correction" << endl;
      if (performance and step%probestride==0)  start_correction = high_resolution_clock::now();
      k=0;
      for (unsigned j=0; j<n_atoms; j++)
      {
        if (!dxnonull[j])
           continue;
        if (dumpderivatives and step%probestride==0)
        {
          ofstream wfile;
          wfile.open("derivatives.csv",std::ios_base::app);
          wfile << setprecision(16);
          wfile << step << " " << j << " " 
                << d_Psi_dx[j] << " " << d_Psi_dy[j] << " " << d_Psi_dz[j] << " "
                << as_scalar(A.row(3).col(k)) << " " << as_scalar(A.row(4).col(k)) << " " << as_scalar(A.row(5).col(k)) << " " 
                << c[k] << endl;      
          wfile.close();          
        }
        d_Psi_dx[j]*=c[k];
        d_Psi_dy[j]*=c[k];
        d_Psi_dz[j]*=c[k];
        tx[j]*=c[k];
        ty[j]*=c[k];
        tz[j]*=c[k];
        k++;
      }
       if (performance and step%probestride==0)  end_correction = high_resolution_clock::now();

      //cout << "checking that correction worked" << endl;
      if (performance and step%probestride==0)  start_test = high_resolution_clock::now();
      for (unsigned j=0; j<n_atoms; j++)
      {
        sum_d_dx+=d_Psi_dx[j];
        sum_d_dy+=d_Psi_dy[j];
        sum_d_dz+=d_Psi_dz[j];
        sum_t_dx+=tx[j];
        sum_t_dy+=ty[j];
        sum_t_dz+=tz[j];
      }

      if ((sum_d_dx>err_tol) or (sum_d_dy>err_tol) or (sum_d_dz>err_tol) or 
          (sum_t_dx>err_tol) or (sum_t_dy>err_tol) or (sum_t_dz>err_tol))
      {
      cout << "Error: Correction of derivatives malfunctioned. Simulation will now end." << endl;
      cout << "Sum derivatives: " << sum_d_dx << " " << sum_d_dy << " " << sum_d_dz << endl;
      cout << "Sum torques: " << sum_t_dx << " " << sum_t_dy << " " << sum_t_dz << endl;
      exit(0);
      }
      if (performance and step%probestride==0)  end_test = high_resolution_clock::now();

      if (performance and step%probestride==0)  end_dxfix = high_resolution_clock::now();
      if (performance and step%probestride==0)
      {
        int tor_time = duration_cast<microseconds>(end_tor - start_tor).count();
        int A_time = duration_cast<microseconds>(end_A - start_A).count();
        int B_time = duration_cast<microseconds>(end_B - start_B).count();
        string Bcoot_time="NA";
        int Bt_time = duration_cast<microseconds>(end_Bt - start_Bt).count();
        int c_time = duration_cast<microseconds>(end_c - start_c).count();
        int correction_time = duration_cast<microseconds>(end_correction - start_correction).count();
        int test_time = duration_cast<microseconds>(end_test - start_test).count();
        int total = duration_cast<microseconds>(end_dxfix - start_dxfix).count();
        ofstream wfile;
        wfile.open("performance_dxfix.txt",std::ios_base::app);
        wfile << tor_time << " " << A_time << " " << B_time << " " << Bcoot_time  << " "<< Bt_time << " " << c_time << " " << correction_time << " " << test_time << " " << total << endl;
        wfile.close();
      }
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
      if (!probe_init_pdb.empty()) // place probes on the coordinates of an input ligand
      {
       x=probe_init.getPositions()[0][0];
       y=probe_init.getPositions()[0][1];
       z=probe_init.getPositions()[0][2];
       probe.place_probe(x,y,z);   
      }
      else 
      {
       unsigned init_j=aidefunctions::get_random_integer(0,n_atoms);
       double x=getPosition(init_j)[0];
       double y=getPosition(init_j)[1];
       double z=getPosition(init_j)[2];
       probe.place_probe(x,y,z);
       probe.perturb_probe(0);
       cout << "Probe " << probe.probe_id << " centered on atom " << atoms[init_j].serial() << endl;
      }
      return;
    }
    // calculator
    void Ghostprobe::calculate()
    {
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
      
      if (performance and step%probestride==0) start_psi = high_resolution_clock::now();
         
      // Update probe coordinates
      if (!noupdate)
      {
       probe.move_probe(step, atoms_x, atoms_y, atoms_z);
      }

      //Calculate Psi and its derivatives
      if (!nocvcalc)
      {
        probe.calculate_activity(atoms_x, atoms_y, atoms_z);
        Psi = probe.activity;
        d_Psi_dx = probe.d_activity_dx;
        d_Psi_dy = probe.d_activity_dy;
        d_Psi_dz = probe.d_activity_dz;
      }

      // print output files
      if (step%probestride==0)
      {
        probe.print_probe_xyz(step);
        probe.print_probe_movement(step, atoms, n_atoms);
        if (protein_out)
        {
         print_protein();
        }
      }
       
      /*
      perturb_probe() needs to go after calculation of activity
      if we are using the derivatives to define the direction of the perturbation.
      If not, we can put it at the top, where it might make a bit more sense.
      */
      if (kpert>0)
      {
        probe.perturb_probe(step);
      }
      
      if (performance and step%probestride==0)  end_psi = high_resolution_clock::now();
      
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
      
      if (performance and step%probestride==0)
      {
       int psi_time = duration_cast<microseconds>(end_psi - start_psi).count();
       int dxfix_time = duration_cast<microseconds>(end_dxfix - start_dxfix).count();
       ofstream wfile;
       wfile.open("performance.txt",std::ios_base::app);
       wfile << psi_time << " " << dxfix_time << endl;
       wfile.close();
      }

      // if (step>=10) exit(0);

    } // close calculate
  }   // close colvar
} // close plmd
