#!/usr/bin/env bash
#SBATCH -A C3SE2016-1-11
#SBATCH -p glenn
#SBATCH -J test
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o test.stdout
#SBATCH -e test.stderr

module purge

module load gcc/4.9/4.9.2
export PYTHONPATH="${PYTHONPATH}:/c3se/users/lyaa/Glenn/xgboost/python-package"
export PATH="/c3se/users/lyaa/Hebbe/miniconda2/bin:$PATH"

pdcp allstate1.py $TMPDIR
pdcp *.csv $TMPDIR

cd $TMPDIR
##
python allstate1.py

cp * $SLURM_SUBMIT_DIR

# End script
