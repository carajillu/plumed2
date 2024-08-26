start=$1
end=$2
plumed_srcdir=/home/joan/github/plumed2_carajillu/
backupdir=/home/joan/MD/GHOSTPROBE/TEM1/1JWP/backup
rootdir=$(pwd)
gmx=/home/joan/local/local/gromacs2023.3-plumed2.10-dev/bin/gmx_mpi
mdpdir=/home/joan/MD/mdp
topname=1JWP.top
plumedat=$3
for i in $(seq $start $end); do
    cp -r $backupdir prod_$i
    cd prod_$i
    if [ $i -gt 0 ]; then
       rm prod.tpr
       sed -i "s/NPROBES=16 PROBESTRIDE=25000/NPROBES=16 PROBESTRIDE=25000 RESTART_PROBES/g" $plumedat
       let a=i-1
       python3 $plumed_srcdir/src/GHOSTPROBE/python/restart.py -d ../prod_$a -n 16 
       $gmx grompp -f $mdpdir/continuation.mdp -c ../prod_$a/prod.gro -p $topname -o prod.tpr
    fi
    $gmx mdrun -v -ntomp 8 -pin on -deffnm prod -plumed $plumedat -nsteps 25000000
    cd $rootdir 	
done
