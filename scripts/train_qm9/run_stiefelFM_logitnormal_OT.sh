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

CKPT_DIR=${SLURM_SUBMIT_DIR}/ckpt/stiefelFM_logitnormal_OT

srun --nodes=1 --tasks-per-node=4 python -m src.experimental.train --enable_wandb --wandb_project=train_kreed_moments_flow --enable_checkpoint --num_nodes=1 --accelerator=gpu --devices=4 --num_workers=15 --batch_size_train=256 --batch_size_eval=100 --max_epochs=1000 --check_samples_every_n_epochs=100 --cond_features=512 --hidden_features=768 --num_heads=12 --num_layers=16 --samples_assess_n_batches=3 --log_every_n_steps=10 --lr=1e-4 --ema_decay=0.9995 --checkpoint_dir=$CKPT_DIR --resume_path=$CKPT_DIR/last.ckpt --model=flow --timesteps=200 --flow_sampling=logitnormal_0.0_1.0 --flow_OT_repeats=3 --flow_OT_limit=50  --wandb_run_name=qm9:stiefelFM_logitnormal_OT
