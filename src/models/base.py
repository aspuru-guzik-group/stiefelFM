import abc

import torch.nn as nn

from src.chem import Molecule


class GenerativeModel(abc.ABC, nn.Module):

    def __init__(self, dynamics: nn.Module, timesteps):
        super().__init__()

        self.dynamics = dynamics
        self.T = timesteps

    def wrap_dataset(self, dataset):
        raise NotImplementedError()

    def objective(self, M: Molecule, **kwargs):
        raise NotImplementedError()

    def sample(self, M: Molecule, pbar=False, return_trajectory=False, **kwargs):
        raise NotImplementedError()
