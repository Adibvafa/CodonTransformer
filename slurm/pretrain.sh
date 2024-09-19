#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=your_output_directory/output_%j.out
#SBATCH --error=your_error_directory/error_%j.err
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=23:59:00
#SBATCH -p compute_full_node

# Load required modules
module --ignore_cache load cuda/11.4.4
module --ignore_cache load anaconda3
source activate your_environment

# Change to the working directory
cd your_working_directory

# Set environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Run the Python script with arguments
stdbuf -oL -eL srun python pretrain.py \
    --tokenizer_path your_tokenizer_path/CodonTransformerTokenizer.json \
    --train_data_path your_data_directory/pretrain_dataset.json \
    --checkpoint_dir your_checkpoint_directory \
    --batch_size 6 \
    --max_epochs 5 \
    --num_workers 5 \
    --accumulate_grad_batches 1 \
    --num_gpus 16 \
    --learning_rate 0.00005 \
    --warmup_fraction 0.1 \
    --save_interval 5 \
    --seed 123
