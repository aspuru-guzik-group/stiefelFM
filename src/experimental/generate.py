import datetime
from typing import List, Literal, Optional

import pydantic_cli
import pydantic
import torch
import numpy as np
import os
import tqdm
import pathlib
import time

from src.chem import Molecule
import torch_geometric as pyg
from src.metrics import evaluate_prediction, compute_curve_length, compute_diversity
from src.datamodule import ConformerDataModule
from src.experimental.lit import LitGenConfig, LitGen

class GenConfig(pydantic.BaseModel):

    ckpt_path: str
    dataset: Literal["qm9", "geom"] = "qm9"

    partition_idx: int
    chunksize: int # number of examples to generate samples for at once
    K: int = 10 # number of samples per example
    dps: bool = False
    timesteps: int
    stochastic_gamma: Optional[float] = None
    
    compute_curve_length: bool = False

    save_path: str

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def generate(config: GenConfig):
    cfg = config

    if cfg.dataset == "qm9":
        dm = ConformerDataModule("qm9", batch_size_train=10, num_workers=0)
        PARTITIONSIZE = 1100
    elif cfg.dataset == "geom":
        dm = ConformerDataModule("geom", batch_size_train=10, num_workers=0, only_lowest_energy_conformers=True)
        PARTITIONSIZE = 920
    
    pbar = False

    # ### TODO: delete this
    # PARTITIONSIZE = 2
    # pbar = True
    # ###

    save_path = pathlib.Path(cfg.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.exists():
        print(f"{save_path} already exists")
        return 1
    
    dset = dm.datasets["test"]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    lit = LitGen.load_from_checkpoint(cfg.ckpt_path, map_location=device)
    gen = lit.ema.ema_model
    gen = gen.to(torch.float64)
    gen.eval();
    
    # Partition the dataset
    test_set_idxs = np.arange(len(dset))
    partitions = [test_set_idxs[i:i+PARTITIONSIZE] for i in range(0, len(test_set_idxs), PARTITIONSIZE)]

    print(f"handling partition {cfg.partition_idx} of {len(partitions)}")
    partition = partitions[cfg.partition_idx]

    chunks = [partition[i:i+cfg.chunksize] for i in range(0, len(partition), cfg.chunksize)]
    print([len(c) for c in chunks])


    # Generate samples
    all_results = dict()
    # structure: example_idx -> [list of K sample_metrics dicts]

    times = []
    for chunk in tqdm.tqdm(chunks):
        Ms = []
        for idx in chunk:
            M = dset[idx]
            Ms.extend([M for _ in range(cfg.K)])
    
        M_batched = pyg.data.Batch.from_data_list(Ms)
        M_batched.batch = torch.repeat_interleave(torch.arange(len(Ms)), M_batched.sizes)
        
        M_batched.coords = M_batched.coords.to(torch.float64)
        M_batched = M_batched.to(device)
        kwargs = {
            "M": M_batched,
            "pbar": pbar,
            "return_trajectory": cfg.compute_curve_length,
            "timesteps": cfg.timesteps,
            "dps": 1.0 if cfg.dps else None,
            "stochastic_gamma": cfg.stochastic_gamma,
        }
        
        start = time.time()
        out = gen.sample(**kwargs)
        gen_time = time.time() - start
        print("time to generate samples:", gen_time)
        times.append(gen_time)

        if cfg.compute_curve_length:
            all_samples, batched_trajs = out
            # all_trajs is a list of batched molecules, of length T

            all_trajs = []
            for j in range(len(all_samples)):
                all_trajs.append([])
            for batched in batched_trajs:
                for j in range(len(all_samples)):
                    all_trajs[j].append(batched[j].coords)

            # all_trajs is a list of single sample trajectories, only coords
        else:
            all_samples = out

        
        for i, idx in enumerate(chunk):
            # iterates over examples in a chunk

            start = cfg.K * i
            end = cfg.K * (i + 1)
            samples = all_samples[start:end]

            if cfg.compute_curve_length:
                trajs = all_trajs[start:end]
            
            M_true = dset[idx]
            M_true.true_coords = M_true.coords

            preds = []
            for j, M_pred in enumerate(samples):
                metrics, M_aligned = evaluate_prediction(
                    M_pred=M_pred,
                    M_true=M_true,
                    return_aligned=True,
                )

                metrics["coords"] = M_aligned.coords

                if cfg.compute_curve_length:
                    # metrics["traj"] = trajs[j]
                    metrics["curve_length"] = compute_curve_length(M_pred, trajs[j])
                
                preds.append(metrics)
            
            # compute diversity for this example
            diversity = compute_diversity(M_true.atoms, [p["coords"] for p in preds])

            for p in preds:
                p["diversity"] = diversity
        
            all_results[idx] = preds
    
    print("average time to generate samples:", np.mean(times))

    torch.save(all_results, save_path)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(GenConfig, generate)
