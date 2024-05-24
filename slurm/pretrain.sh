#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=/scratch/g/gfilion/adibvafa/Codon/output_%j.out
#SBATCH --error=/scratch/g/gfilion/adibvafa/Codon/error_%j.err
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=23:59:00
#SBATCH -p compute_full_node

module --ignore_cache load cuda/11.4.4
module --ignore_cache load anaconda3
source activate light

cd /scratch/g/gfilion/adibvafa/Codon

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 pretrain.py
