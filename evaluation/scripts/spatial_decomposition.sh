#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=14
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=200GB
#SBATCH --account=kainmueller
#SBATCH --nodelist=maxg[10,20]
#SBATCH --output=/fast/AG_Kainmueller/vguarin/aggrigator_experiments/log_%j.out
#SBATCH --error=/fast/AG_Kainmueller/vguarin/aggrigator_experiments/log_%j.err
#SBATCH --export=ALL
#SBATCH --partition=h100
#SBATCH -pkainmueller

echo "Job ID: $SLURM_JOB_ID"

python evaluation/scripts/evaluate_spatial_fingerprint.py --dataset 'gta' --uq_method 'softmax'
