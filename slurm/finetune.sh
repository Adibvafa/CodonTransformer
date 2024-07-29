#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=your_output_directory/output_%j.out
#SBATCH --error=your_error_directory/error_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --time=15:00:00
#SBATCH --partition=compute_full_node

# Load required modules
module --ignore_cache load cuda/11.4.4
module --ignore_cache load anaconda3
source activate light

# Change to the working directory
cd your_working_directory

# Set environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Run the Python script with arguments
stdbuf -oL -eL srun python finetune.py \
    --dataset_dir your_dataset_directory \
    --checkpoint_dir your_checkpoint_directory \
    --checkpoint_filename finetune.ckpt \
    --batch_size 6 \
    --max_epochs 15 \
    --num_workers 5 \
    --accumulate_grad_batches 1 \
    --num_gpus 4 \
    --learning_rate 0.00005 \
    --warmup_fraction 0.1 \
    --save_every_n_steps 512 \
    --seed 123