#!/usr/bin/env bash
#SBATCH -A C3SE2016-1-11
#SBATCH -p hebbe
#SBATCH -J BG0 
#SBATCH -N 1
#SBATCH -C GPU
#SBATCH -t 120:00:00
#SBATCH -o BG0.stdout 
#SBATCH -e BG0.stderr 

module purge
module load CUDA
export PYTHONPATH="${PYTHONPATH}:/c3se/NOBACKUP/users/lyaa/Hebbe/xgboost/python-package"
export PATH="/c3se/NOBACKUP/users/lyaa/Hebbe/miniconda3/bin:$PATH"

pdcp allstate1.py $TMPDIR
pdcp BG0.py $TMPDIR
pdcp {train,test,sample_submission}.csv $TMPDIR

cd $TMPDIR
##
python BG0.py

cp *.pkl $SLURM_SUBMIT_DIR

# End script
