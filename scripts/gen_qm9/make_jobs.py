import numpy as np


# These paths are where the checkpoints are located after training
root = "../train_qm9"
tags_ckpt_paths_timesteps_curve_dps = [
    ("kreedXL", f"{root}/ckpt/kreedXL/last.ckpt", 1000, "", ""),
    ("kreedXL_dps", f"{root}/ckpt/kreedXL/last.ckpt", 1000, "", "--enable_dps"),
    ("stiefelFM", f"{root}/ckpt/stiefelFM/last.ckpt", 200, "--enable_compute_curve_length", ""),
    ("stiefelFM_OT", f"{root}/ckpt/stiefelFM_OT/last.ckpt", 200, "--enable_compute_curve_length", ""),
    ("stiefelFM_logitnormal", f"{root}/ckpt/stiefelFM_logitnormal/last.ckpt", 200, "--enable_compute_curve_length", ""),
    ("stiefelFM_logitnormal_OT", f"{root}/ckpt/stiefelFM_logitnormal_OT/last.ckpt", 200, "--enable_compute_curve_length", ""),
    ("stiefelFM_stoch10", f"{root}/ckpt/stiefelFM_stoch10/last.ckpt", 200, "--enable_compute_curve_length", ""),
]

# get parent directory of this file
import os
import pathlib
dirname = pathlib.Path(os.path.dirname(__file__))
samples_dir = dirname / "samples"

base = "python -m src.experimental.generate --dataset=qm9 --chunksize=15"

jobs = []

n_partitions_to_sample = 12 # max is 12

for tag, ckpt_path, timesteps, curve, dps in tags_ckpt_paths_timesteps_curve_dps:
    ckpt_path = f"{os.path.realpath(ckpt_path)}"
    for i in range(n_partitions_to_sample):
        save_path = samples_dir / tag / f"qm9_{i}.pt"
        job = f"{base} --ckpt_path={ckpt_path} --timesteps={timesteps} {curve} {dps} --partition_idx={i} --save_path={save_path}"
        jobs.append(job)

run_sh = r"""#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:59:0
#SBATCH --job-name=sweep
#SBATCH --array=1-
#SBATCH --output=%A_%a.out

source ~/.bashrc
conda activate moment

cd ../..

eval $(sed -n ${SLURM_ARRAY_TASK_ID}p ${SLURM_SUBMIT_DIR}/array_jobs)
"""

# replace --array=1- with --array=1-{len(jobs)}
run_sh = run_sh.replace("--array=1-", f"--array=1-{len(jobs)}")

with open(f"array_jobs", "w") as f:
    for job in jobs:
        print(job, file=f)

with open(f"run.sh", "w") as f:
    print(run_sh, file=f)
