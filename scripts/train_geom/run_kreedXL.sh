#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=compute_full_node
#SBATCH --time=23:59:0
#SBATCH --job-name=sweep
#SBATCH --output=%j.out

source ~/.bashrc
conda activate moment

cd ../..

wandb offline

CKPT_DIR=${SLURM_SUBMIT_DIR}/ckpt/kreedXL

srun --nodes=1 --tasks-per-node=4 python -m src.experimental.train --enable_wandb --wandb_project=train_kreed_moments_flow --enable_checkpoint --num_nodes=1 --accelerator=gpu --devices=4 --num_workers=15 --batch_size_train=24 --batch_size_eval=32 --check_samples_every_n_epochs=1 --cond_features=512 --hidden_features=768 --num_heads=12 --num_layers=16 --samples_assess_n_batches=5 --log_every_n_steps=10 --lr=1e-4 --ema_decay=0.9995 --dataset=geom --model=ddpm --timesteps=1000 --disable_ddpm_predict_eps --ddpm_dps=1.0 --enable_ddpm_project --wandb_run_name=geom:kreedXL --checkpoint_dir=$CKPT_DIR --resume_path=$CKPT_DIR/last.ckpt --max_epochs=60