import pathlib

import joblib
import lightning.pytorch as pl
import numpy as np
import torch
import torch_geometric as pyg
import tqdm

from src.chem import Molecule
from src.paths import DATA_ROOT

RAW_FILE_URLS = {
    "qm9": (
        "qm9_processed.tar.gz",
        "https://drive.google.com/u/0/uc?id=1jmc2JBoXJxat_Aq74E3ffCIQGKH9JuG-&confirm=t",
    ),
    "geom": (
        "geom_processed.tar.gz",
        "https://drive.google.com/u/0/uc?id=1UXDaJak686jtEyyfJrTxkiOkYT1SsKyK&confirm=t",
    ),
}

BAD_QM9_EXAMPLES = np.array([126595, 17167, 56848, 45895, 18183, 119102, 107249])
QM9_SMALL_MOMENTS = np.load(DATA_ROOT / "splits" / "qm9_small_moments.npy")
QM9_TOO_SMALL = np.load(DATA_ROOT / "splits" / "qm9_too_small.npy")

MIN_ATOMS = 5
SPLITS = ["train", "val", "test"]


def make_molecule(info, axyz):
    _, molecule_id, geom_id = list(info)

    # Load molecule
    axyz = torch.from_numpy(axyz)
    atoms, coords = axyz[:, :1].long(), axyz[:, 1:]
    size = coords.shape[0]
    assert size > 1
    if size < MIN_ATOMS:
        return None
    sizes = torch.tensor([size], dtype=torch.long)

    M = Molecule(
        coords=coords,
        atoms=atoms,
        id=geom_id,
        sizes=sizes,
        num_nodes=size,
    )
    M.validate()

    coords, moments = M.rotated_to_principal_axes(return_moments=True)
    M.coords = coords
    M.moments = moments
    return M


class ConformerDataset(pyg.data.InMemoryDataset):

    def __init__(
        self,
        dataset,
        split,
        root=str(DATA_ROOT),
        only_lowest_energy_conformers=False,
        num_process_jobs=12,
    ):
        self.metadata = {}
        self.dataset = dataset
        self.split = split
        self.only_lowest_energy_conformers = only_lowest_energy_conformers
        self.num_process_jobs = num_process_jobs

        super().__init__(root=root)  # (!!!) order here is actually important
        self.data, self.slices = torch.load(self.processed_paths[SPLITS.index(split)])

    @property
    def raw_file_names(self):
        return [f"{self.dataset}/processed/coords.npy", f"{self.dataset}/processed/metadata.npy"]

    @property
    def processed_file_names(self):
        if self.only_lowest_energy_conformers:
            assert self.dataset == "geom"
            return [f"{self.dataset}lowest_{split}.pyg" for split in SPLITS]

        return [f"{self.dataset}_{split}.pyg" for split in SPLITS]

    # def download(self):
    #     import gdown
    #     import tarfile
    #     fname, url = RAW_FILE_URLS[self.dataset]
    #     gdown.download(url, f"{self.raw_dir}/{fname}", quiet=False)

    #     # extract tar.gz
    #     with tarfile.open(f"{self.raw_dir}/{fname}", "r:gz") as tar:
    #         tar.extractall(path=self.raw_dir)

    # def process(self):

    #     # Load data
    #     data_dir = pathlib.Path(__file__).parents[1] / "data" / self.raw_dir / self.dataset / "processed"
    #     metadata = np.load(str(data_dir / "metadata.npy"))  # each row is (start_index, molecule_id, geom_id)
    #     coords = np.load(str(data_dir / "coords.npy"))

    #     # Unbind coordinates
    #     start_indices = metadata[:, 0]
    #     all_coords = np.split(coords, start_indices[1:])

    #     for split, path in zip(SPLITS, self.processed_paths):
    #         if self.only_lowest_energy_conformers:
    #             assert self.dataset == "geom"
    #             idxs_path = DATA_ROOT / "splits" / f"{self.dataset}lowest_{split}.npy"
    #         else:
    #             idxs_path = DATA_ROOT / "splits" / f"{self.dataset}_{split}.npy"

    #         idxs = np.load(str(idxs_path))
    #         idxs = np.sort(idxs)
    #         if self.dataset == "qm9":
    #             idxs = np.setdiff1d(idxs, BAD_QM9_EXAMPLES)
    #             idxs = np.setdiff1d(idxs, QM9_SMALL_MOMENTS)
    #             idxs = np.setdiff1d(idxs, QM9_TOO_SMALL)
    #             # idxs = list(range(10))

    #         conformers = joblib.Parallel(n_jobs=self.num_process_jobs, backend="loky")(
    #             joblib.delayed(make_molecule)(metadata[idx], all_coords[idx])
    #             for idx in tqdm.tqdm(idxs, desc=f"Processing {split} split")
    #         )
    #         conformers = [c for c in conformers if c is not None]
    #         # import copy
    #         # conformers = [copy.copy(c) for _ in range(100_000) for c in conformers]

    #         data, slices = self.collate(conformers)
    #         torch.save((data, slices), path)


class ConformerDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset,
        root=str(DATA_ROOT),
        batch_size_train=10,
        batch_size_eval=10,
        num_workers=0,
        only_lowest_energy_conformers=False,
        wrapper=None,
        num_process_jobs=12,
    ):
        super().__init__()

        self.batch_sizes = {
            "train": batch_size_train,
            "val": batch_size_eval,
            "test": batch_size_eval,
        }
        self.num_workers = num_workers
        self.wrapper = wrapper

        self.datasets = {
            split: ConformerDataset(
                dataset=dataset,
                split=split,
                only_lowest_energy_conformers=only_lowest_energy_conformers,
                root=root,
                num_process_jobs=num_process_jobs,
            )
            for split in SPLITS
        }

    def train_dataloader(self):
        return self._loader(split="train", shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(split="val", shuffle=True)

    def test_dataloader(self):
        return self._loader(split="test")

    def _loader(self, split, shuffle=False, drop_last=False):
        dataset = self.datasets[split]
        if self.wrapper is not None:
            dataset = self.wrapper(dataset)
        collate_fn = dataset.collate

        return pyg.loader.DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_sizes[split],
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
