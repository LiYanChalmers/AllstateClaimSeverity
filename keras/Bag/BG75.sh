#!/usr/bin/env bash
#SBATCH -A C3SE2016-1-11
#SBATCH -p glenn
#SBATCH -J BG75 
#SBATCH -N 1
#SBATCH -t 100:00:00
#SBATCH -o BG75.stdout 
#SBATCH -e BG75.stderr 

module purge

export PYTHONPATH="${PYTHONPATH}:/c3se/NOBACKUP/users/lyaa/Hebbe/xgboost/python-package"
export PATH="/c3se/NOBACKUP/users/lyaa/Hebbe/miniconda3/bin:$PATH"

pdcp allstate1.py $TMPDIR
pdcp BG75.py $TMPDIR
pdcp ../allstate/{train,test,sample_submission}.csv $TMPDIR

cd $TMPDIR
##
python BG75.py

cp * $SLURM_SUBMIT_DIR

# End script
