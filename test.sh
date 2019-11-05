#!/bin/sh
#PBS -q l-small
#PBS -l select=1:mpiprocs=1
#PBS -Wgroup_list=gk43
#PBS -l walltime=02:00:00
#PBS -e err
#PBS -o 256_8.txt

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh
export I_MPI_PIN_DOMAIN=socket
export I_MPI_PERHOST=2
export MODULEPATH=/lustre/gk43/k43003/conda-envs:$MODULEPATH
module load 2.0.0

export CONDA_ENVS_PATH=/lustre/gk43/k43003/conda-envs
source activate TF2rc
python NL_PendulumExperiment.py --epoch 5 --experiment baka --gpu 1
