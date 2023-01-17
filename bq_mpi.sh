#!/bin/bash
#SBATCH --partition=generalsky
##SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --mail-type=END,ARRAY_TASKS
#SBATCH --mail-user=andrew.casey-clyde@uconn.edu
#SBATCH --output=bq_mpi.out
#SBATCH --array=0-59%1

module purge
module load intel/2019u3
module load java/1.8.0_162
module load zlib/1.2.11
##module load gcc/9.4.0
module load mpi/openmpi
##module load anaconda/5.1.0
##modp mpi/openmpi
##module load mpi/mvapich2

source /home/jac19017/miniconda3/etc/profile.d/conda.sh
conda activate qso_signposts
mpiexec python3 qlf_fit.py --quasars ./data/processed/sky_freq_mock_bq_pop_incomplete.csv --track_acorr  --filename bq_mpi --config ./bq_mpi.yaml --n_samples 1000 --n_walkers 192 --n_cpus 192 --crts_based --extensions --cont --mpi
