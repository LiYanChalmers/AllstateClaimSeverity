#!/usr/bin/env bash
#SBATCH -A C3SE2016-1-11
#SBATCH -p glenn
#SBATCH -J HOT91 
#SBATCH -N 1
#SBATCH -t 48:10:00
#SBATCH -o HOT91.stdout 
#SBATCH -e HOT91.stderr 

module purge

module load gcc/4.9/4.9.2
export PYTHONPATH="${PYTHONPATH}:/c3se/users/lyaa/Glenn/xgboost/python-package"
export PATH="/c3se/users/lyaa/Hebbe/miniconda2/bin:$PATH"

pdcp allstate1.py $TMPDIR
pdcp allstate2HOCluster91.py $TMPDIR
pdcp train.csv test.csv sample_submission.csv $TMPDIR
pdcp train_test_encoded.pkl parameterList.pkl $TMPDIR
pdcp selected_categorical_features.pkl $TMPDIR
pdcp train_test_encoded_xgb150_pairs35.pkl $TMPDIR

cd $TMPDIR
##
python allstate2HOCluster91.py

cp * $SLURM_SUBMIT_DIR

# End script
