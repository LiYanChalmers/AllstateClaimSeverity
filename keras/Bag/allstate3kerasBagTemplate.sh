#!/usr/bin/env bash
#SBATCH -A C3SE2016-1-11
#SBATCH -p glenn
#SBATCH -J HOTemplate
#SBATCH -N 1
#SBATCH -t 100:00:00
#SBATCH -o HOTemplate.stdout
#SBATCH -e HOTemplate.stderr

module purge

export PYTHONPATH="${PYTHONPATH}:/c3se/NOBACKUP/users/lyaa/Hebbe/xgboost/python-package"
export PATH="/c3se/NOBACKUP/users/lyaa/Hebbe/miniconda3/bin:$PATH"


pdcp allstate2HOClusterTemplate.py $TMPDIR
pdcp ../allstate/{train,test,sample_submission}.csv $TMPDIR

cd $TMPDIR
##
python allstate2HOClusterTemplate.py

cp * $SLURM_SUBMIT_DIR

# End script
