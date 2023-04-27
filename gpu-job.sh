#!/bin/bash -l

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

#SBATCH --time=6:00:00 

#SBATCH -A plgexaile-gpu-a100
#SBATCH -p plgrid-gpu-a100

conda activate abbr

srun python train.py --do_train --do_eval
