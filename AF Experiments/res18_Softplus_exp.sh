#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --job-name=res18_Softplus_exp
#SBATCH --mem=8000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --output=res18_Softplus_exp.txt

module load Python

python3 res18_Softplus_exp.py