#!/bin/bash -l

#SBATCH -J sweep
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --time=6:00:00 

#SBATCH -A plgexaile-gpu-a100
#SBATCH -p plgrid-gpu-a100

cd $SCRATCH/abbr

conda activate abbr

srun wandb agent --count 1 carbon-agh/poleval-2022-abbr/XXXX
