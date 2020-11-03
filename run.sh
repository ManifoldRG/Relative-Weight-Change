#!/bin/sh
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_v100'
#SBATCH --job-name=mnist_resnet_baseline
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=atendle13.3.98@gmail.com
#SBATCH --output=/work/netthinker/atendle/MC/mnist_res_baseline.out

export PYTHONPATH=$WORK/tf-gpu-pkgs
module load singularity
singularity exec docker://lordvoldemort28/pytorch-opencv:dev python -u $@
