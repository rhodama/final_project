#!/bin/bash
#SBATCH --account=m4776
#SBATCH -C gpu
#SBATCH --gpus 1
#SBATCH --qos=shared
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH -n 1




srun ./build/gpu -i input.txt -o ./results -g 10000

