#!/usr/bin/env bash
#SBATCH -A C3SE2016-1-11
#SBATCH -p glenn
#SBATCH -J KF25 
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -o KF25.stdout 
#SBATCH -e KF25.stderr 

module purge

export PYTHONPATH="${PYTHONPATH}:/c3se/NOBACKUP/users/lyaa/Hebbe/xgboost/python-package"
export PATH="/c3se/NOBACKUP/users/lyaa/Hebbe/miniconda3/bin:$PATH"

pdcp allstate1.py $TMPDIR
pdcp allstate3kerasGlenn25.py $TMPDIR
pdcp input_keras.pkl $TMPDIR

cd $TMPDIR
##
python allstate3kerasGlenn25.py

cp * $SLURM_SUBMIT_DIR

# End script
