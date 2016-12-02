#!/usr/bin/env bash
#SBATCH -A C3SE407-15-3
#SBATCH -p hebbe
#SBATCH -J HOTemplate
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 48:00:00
#SBATCH -o HOTemplate.stdout
#SBATCH -e HOTemplate.stderr

module purge


export PYTHONPATH="${PYTHONPATH}:/c3se/users/lyaa/Hebbe/xgboost/python-package"
export PATH="/c3se/users/lyaa/Hebbe/miniconda3/bin:$PATH"

pdcp allstate1.py $TMPDIR
pdcp allstate2HOClusterTemplate.py $TMPDIR
pdcp /c3se/users/lyaa/Glenn/allstate/train.csv /c3se/users/lyaa/Glenn/allstate/test.csv /c3se/users/lyaa/Glenn/allstate/sample_submission.csv $TMPDIR
pdcp /c3se/users/lyaa/Glenn/allstate/train_test_encoded.pkl $TMPDIR
pdcp /c3se/users/lyaa/Glenn/allstate/selected_categorical_features.pkl $TMPDIR
pdcp /c3se/users/lyaa/Glenn/allstate/train_test_encoded_xgb150_pairs35.pkl $TMPDIR

cd $TMPDIR
##
python allstate2HOClusterTemplate.py

cp * $SLURM_SUBMIT_DIR

# End script
