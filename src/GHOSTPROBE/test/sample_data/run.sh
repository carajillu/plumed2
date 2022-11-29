# Generate the plumed.dat file including only protein heavy atoms
python ../../python/make_plumedat.py -f equil.gro -s protein

# Run and postprocess 25 simulations of 10 ns
for sim in {0..4}; do
    mkdir prod_$sim; 
    cp prod.tpr* plumed.dat* prod_$sim/; 
    cd prod_$sim; 
       gmx mdrun -v -update gpu -ntomp 16 -deffnm prod -nsteps 0 -plumed plumed.dat # 10 ns
       gmx trjconv -f  prod.xtc -s prod.tpr -pbc mol -ur compact -center -o prod_dry.xtc << eof
1
1
eof
       gmx trjconv -f  prod.xtc -s prod.tpr -pbc mol -ur compact -center -o prod_dry.gro -b 0 -e 0 << eof
1
1
eof
       python ~/github/plumed2_carajillu/src/GHOSTPROBE/python/postprocess.py -i prod_dry.gro -t prod_dry.xtc -x protein.xyz -n 16 -a 1.0 
    cd ..; 
done

#Extract fully active probes
cat */probes_actmin_1.0.pdb | grep ATOM > probes_actmin_1.0.pdb
