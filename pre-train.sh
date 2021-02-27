#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --job-name=pre-train
#SBATCH --mem=4000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --output=pre-train_epoch_info.txt

module load Python/3.6.4-foss-2018a

python3 fine-tune.py