#!/bin/bash
#SBATCH -J serial      
#SBATCH -o serial_%j.out 
#SBATCH -e serial_%j.err 
#SBATCH -A m4776             
#SBATCH -C cpu               
#SBATCH -c 1                 
#SBATCH --qos=debug          
#SBATCH -t 00:30:00          
#SBATCH -N 1                 
#SBATCH -n 1                 

srun ./build/serial -i input.txt -o ./results -g 10000