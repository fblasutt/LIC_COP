#!/bin/bash

#SBATCH -N 1                 ## number of nodes
#SBATCH --cpus-per-task=20                 ## number of nodes
#SBATCH --mem-per-cpu=10GB            ## number of nodes
#SBATCH -t 0:12:00          ## walltime
#SBATCH	--job-name="py_ivf"    ## name of job

module load releases/2023b
module load Python/3.11.5-GCCcore-13.2.0

cd /home/users/b/l/blasutto/LIC_COP/



python -u time.py


exit