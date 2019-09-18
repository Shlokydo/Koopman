#!/bin/sh
#PBS -q h-short
#PBS -l select=1:mpiprocs=1
#PBS -Wgroup_list=gk43
#PBS -l walltime=02:00:00
#PBS -e err
#PBS -o output.lst

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh
export I_MPI_PIN_DOMAIN=socket
export I_MPI_PERHOST=2
export MODULEPATH=/lustre/app/modulefiles/test:$MODULEPATH
module load anaconda3/4.3.0 cuda10 tensorflow/2.0.0a

python ./NL_PendulumExperiment.py
