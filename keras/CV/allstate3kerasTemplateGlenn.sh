#!/usr/bin/env bash
#SBATCH -A C3SE2016-1-11
#SBATCH -p glenn
#SBATCH -J HOTemplate
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -o HOTemplate.stdout
#SBATCH -e HOTemplate.stderr

module purge

export PYTHONPATH="${PYTHONPATH}:/c3se/NOBACKUP/users/lyaa/Hebbe/xgboost/python-package"
export PATH="/c3se/NOBACKUP/users/lyaa/Hebbe/miniconda3/bin:$PATH"

pdcp allstate1.py $TMPDIR
pdcp allstate2HOClusterTemplate.py $TMPDIR
pdcp input_keras.pkl $TMPDIR

cd $TMPDIR
##
python allstate2HOClusterTemplate.py

cp * $SLURM_SUBMIT_DIR

# End script
