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

#srun --export=ALL,LD_PRELOAD=$LIBDARSHAN  #to run with darshan
srun python main.py