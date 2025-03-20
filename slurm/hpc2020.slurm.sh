#!/bin/bash
#SBATCH --job-name=aifs-dataloader-bm
#SBATCH --qos=np
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out

cd /ec/res4/hpcperm/naco/aifs/anemoi-dataloader-microbenchmark
source env.sh
source darshan/enable-darshan

grep -HP 'llite|cgroup' $PWD/config.h

export DARSHAN_LOG_PATH=$DARSHAN_LOG_PATH/slurm-$SLURM_JOBID
mkdir -p $DARSHAN_LOG_PATH
DARSHAN_ENABLE_NONMPI=1 srun python main.py
#srun python main.py

grep -HP 'llite|cgroup' $PWD/endt.h