import pandas as pd
import numpy as np
import torch
import tqdm

all_dicts = dict()

for key in tqdm.tqdm(["stiefelFM", "stiefelFM_OT", "stiefelFM_more1", "stiefelFM_more2", "stiefelFM_OT_more1", "stiefelFM_OT_more2"]):
    all_dicts[key] = torch.load(f"../../samples/geom/{key}.pt")

df = pd.read_csv("geom_results.csv")

parts = []
for ckpt in ["stiefelFM", "stiefelFM_OT"]:
    for i, more in enumerate(["", "_more1", "_more2"]):
        part = df[df.ckpt == ckpt + more].copy()
        part["sample_idx"] += i*10
        part["ckpt"] = ckpt
        parts.append(part)

df2 = pd.concat(parts)

import sys
sys.path.append("../..")
from src.datamodule import ConformerDataModule
from src.metrics import compute_diversity
import torch

geom = ConformerDataModule("geom", batch_size_train=2, num_workers=0, only_lowest_energy_conformers=True)
dset = geom.datasets["test"]

dfrel = df2[df2.ckpt.isin(["stiefelFM", "stiefelFM_OT"])]

dfsorted = dfrel.sort_values(by=["ckpt", "test_set_idx", "validity"], ascending=[True, True, False])
ranked = np.tile(np.arange(30), len(dfrel) // 30)
dfsorted["ranked"] = ranked

out = dfsorted[dfsorted.ranked < 10].copy()

idx_to_folder = {0: "", 1: "_more1", 2: "_more2"}

for ckpt in ["stiefelFM", "stiefelFM_OT"]:
    all_dicts[f"{ckpt}_filter"] = dict()
    for test_set_idx in tqdm.trange(29203):
        example = out[(out.ckpt == ckpt) & (out.test_set_idx == test_set_idx)]
        atoms = dset[test_set_idx].atoms
        coords_list = []

        list_of_dicts = []
        for j, row in example.iterrows():
            folder_idx = row.sample_idx // 10
            folder = idx_to_folder[folder_idx]
            sample_idx = (row.sample_idx % 10)
            coords_list.append(all_dicts[f"{ckpt}{folder}"][test_set_idx][sample_idx]["coords"])

            newdict = all_dicts[f"{ckpt}{folder}"][test_set_idx][sample_idx].copy()
            list_of_dicts.append(newdict)

        diversity = compute_diversity(atoms, coords_list)

        for i in range(len(list_of_dicts)):
            list_of_dicts[i]["diversity"] = diversity
        
        all_dicts[f"{ckpt}_filter"][test_set_idx] = list_of_dicts
    
    torch.save(all_dicts[f"{ckpt}_filter"], f"../../samples/geom/{ckpt}_filter.pt")

