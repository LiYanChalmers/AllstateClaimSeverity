#!/usr/bin/env bash
#SBATCH -A C3SE2016-1-11
#SBATCH -p hebbe
#SBATCH -J sv231 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C "MEM128|MEM64"
#SBATCH -t 20:00:00
#SBATCH -o sv231.stdout 
#SBATCH -e sv231.stderr 

module purge

export PYTHONPATH="${PYTHONPATH}:/c3se/NOBACKUP/users/lyaa/Hebbe/xgboost/python-package"
export PATH="/c3se/NOBACKUP/users/lyaa/Hebbe/miniconda3/bin:$PATH"

pdcp allstate1.py $TMPDIR
pdcp svr231.py $TMPDIR
pdcp param_samples400.pkl input_svr.pkl $TMPDIR

cd $TMPDIR
##
python svr231.py

cp *.pkl $SLURM_SUBMIT_DIR

# End script
