# This script assumes that the samples of KREED-XL have been generated in 32 partitions of the test set of GEOM, as specified in make_jobs.py
# This script will load the samples and project them to the feasible manifold of the true molecule, then evaluate the projected samples, and finally save the results of all partitions together to a single file.

import sys
sys.path.append("../..")
from src.datamodule import ConformerDataModule
import torch
import numpy as np
import tqdm
from src.metrics import evaluate_prediction, compute_diversity

dm = ConformerDataModule("geom", only_lowest_energy_conformers=True)
dset = dm.datasets["test"]

samples_path = f"samples/kreedXL"

all_results = dict()
import tqdm
for k in tqdm.trange(32):
    out = torch.load(f"{samples_path}/geom_{k}.pt")
    for test_set_idx in tqdm.tqdm(out):
        M_true = dset[test_set_idx]
        M_true.true_coords = M_true.coords
        preds = []
        for sample_idx in range(len(out[test_set_idx])):
            M_pred = M_true.clone()
            M_pred.coords = out[test_set_idx][sample_idx]["coords"]
            M_pred.coords = M_pred.project_to_manifold()
            metrics, M_aligned = evaluate_prediction(M_pred, M_true, return_aligned=True)
            metrics["coords"] = M_aligned.coords
            preds.append(metrics)
        
        diversity = compute_diversity(M_true.atoms, [p["coords"] for p in preds])
        for p in preds:
            p["diversity"] = diversity
        
        all_results[test_set_idx] = preds

torch.save(all_results, f"{samples_path}/../kreedXL_proj.pt")
