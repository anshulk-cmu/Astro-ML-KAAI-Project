#!/bin/bash
# Task B HEAVY RUN -- 4-snapshot array (771/743/692/660); 817 skipped (not in mock pool).
# GATED: submit ONLY after STOP-2 greenlight.  Submit with:  sbatch code/apFull.sbatch
#SBATCH -p RM
#SBATCH -A phy200026p
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J apFull
#SBATCH --array=0-3
#SBATCH -o /hildafs/projects/phy200026p/akumar45/logs/apFull_%a.out
#SBATCH -e /hildafs/projects/phy200026p/akumar45/logs/apFull_%a.err

cd /hildafs/projects/phy200026p/akumar45
source env.sh
SNAPS=(771 743 692 660)
S=${SNAPS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID -> snapshot $S on $(hostname) $(date)"
python -u code/apertureMass.py full "$S"
echo "done snapshot $S $(date)"
