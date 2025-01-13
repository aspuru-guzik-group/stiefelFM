import os

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import tqdm

from src.chem import Molecule, PTABLE
from src.models.base import GenerativeModel

from src.stiefel_log import OT_permutation_reflection, Stiefel_Exp, Stiefel_Log_alg, Stiefel_Log_alg_noproj


def eye_like(A, usetorch=False):
    if usetorch:
        return torch.eye(A.shape[0]).to(A)
    else:
        return np.eye(A.shape[0], dtype=A.dtype)


def skew(A):
    return 0.5 * (A - A.T)


def project_tanspace(base, col4, v, usetorch=False):
    # base is (N 3), col4 is (N 1) and v is (N 3)
    ONB4 = torch.cat([base, col4], dim=-1) if usetorch else np.concatenate([base, col4], axis=-1)
    v = base @ skew(base.T @ v) + v - ONB4 @ ONB4.T @ v
    return v # (N 3)

def check_ONB(base, col4):
    assert base.shape[-1] == 3

    error = np.abs(base.T @ col4).max().item()
    assert error <= 1e-3, error

    error = np.abs(base.T @ base - eye_like(base.T)).max().item()
    assert error <= 1e-3, error


def check_ONB_tangent(base, v):
    error = np.abs(base.T @ v + v.T @ base).max().item()
    assert error <= 1e-3, error

    # error = np.abs(v[:, -1]).max().item()
    # assert error <= 1e-5, error


def householder(v, usetorch=False):
    if usetorch:
        return eye_like(v, usetorch=True) - (2 * v * v.T / v.square().sum())
    assert (v.ndim == 2) and (v.shape[-1] == 1)
    return eye_like(v) - (2 * v * v.T / np.square(v).sum())


def sample_ONB_0(shape, col4):
    # based on https://geomstats.github.io/_modules/geomstats/geometry/stiefel.html#Stiefel.random_uniform
    std_normal = np.random.randn(*shape).astype(col4.dtype)
    std_normal_transpose = std_normal.T
    aux = std_normal_transpose @ std_normal
    evals, evecs = np.linalg.eigh(aux)
    inv_sqrt_aux = evecs @ np.diag(1 / np.sqrt(evals)) @ evecs.T
    ONB_0 = std_normal @ inv_sqrt_aux
    ONB_0 = householder(col4) @ householder(ONB_0[:, 3:] + col4) @ ONB_0
    # check_ONB(ONB_0, col4)
    return ONB_0


def canonical_norm(base, v):
    norm2 = np.square(v).sum() - 0.5 * np.square(base.T @ v).sum()
    return np.sqrt(norm2)


def kappa_tilde(t, cutoff, d):
    A = 1 / (cutoff - d)
    denom = 1 / (A * cutoff * (1 - t) ** cutoff) + d / cutoff
    return 1 - (1 / denom)  # this 1 - (stuff) is intentional
    # because we use kappa_tilde via exp-wrap from 0

# (old tanspace, should be more correct)
def old_ode_step(ONB, tangent):
    p = 4

    col4 = ONB[:, 3:]

    # project tangent to tangent space at ONB
    tangent = F.pad(tangent, (0, 1))
    A = ONB.T @ tangent
    tangent = tangent - .5 * ONB @ (A + A.T)

    # stiefel exponential
    A = ONB.T @ tangent
    K = tangent - (ONB @ A)
    Q, R = torch.linalg.qr(K)
    AR = torch.cat([A, -R.T], dim=-1)
    Z = torch.zeros_like(tangent)[..., :p, :p]
    RZ = torch.cat([R, Z], dim=-1)
    block = torch.cat([AR, RZ], dim=-2)
    MN_e = torch.linalg.matrix_exp(block)
    exp = (ONB @ MN_e[..., :p, :p]) + (Q @ MN_e[..., p:, :p])

    ONB = exp

    # project to manifold
    U, _, V = torch.linalg.svd(ONB)
    ONB = U[..., :, :p] @ V

    # rotate to col4
    ONB = householder(col4, usetorch=True) @ householder(ONB[:, 3:] + col4, usetorch=True) @ ONB

    return ONB  # should be on-manifold

def ode_step(base, tangent):
    # assumes base is exactly on manifold
    p = 4

    col4 = base[:, 3:]

    # Project tangent to tangent space at ONB
    tangent = project_tanspace(base=base[:, :3], col4=base[:, 3:], v=tangent, usetorch=True)
    tangent = F.pad(tangent, (0, 1))

    # Stiefel exponential
    A = base.T @ tangent
    K = tangent - (base @ A)
    Q, R = torch.linalg.qr(K)
    AR = torch.cat([A, -R.T], dim=-1)
    Z = torch.zeros_like(tangent)[..., :p, :p]
    RZ = torch.cat([R, Z], dim=-1)
    block = torch.cat([AR, RZ], dim=-2)
    MN_e = torch.linalg.matrix_exp(block)
    base = (base @ MN_e[..., :p, :p]) + (Q @ MN_e[..., p:, :p])

    # project to manifold
    U, _, V = torch.linalg.svd(base)
    base = U[..., :, :p] @ V

    # rotate to col4
    base = householder(col4, usetorch=True) @ householder(base[:, 3:] + col4, usetorch=True) @ base

    return base  # should be on-manifold


vmapped_ode_step = torch.vmap(ode_step)
vmapped_old_ode_step = torch.vmap(old_ode_step)


class FlowMatchingDataset(pyg.data.InMemoryDataset):

    def __init__(
        self,
        dataset,
        sampling="linear",
        distance_dependence="none",
        use_OT=True,
        rejection_cutoff=100.0, # not used
        stochastic_gamma=0.0,
        OT_repeats=5,
        OT_limit=100,
        use_log3=True,
        log_noproj=False,
    ):
        super().__init__()

        self.dataset = dataset
        self.sampling = sampling
        self.distance_dependence = distance_dependence
        self.use_OT = use_OT
        self.rejection_cutoff = rejection_cutoff
        self.stochastic_gamma = stochastic_gamma
        self.OT_repeats = OT_repeats
        self.OT_limit = OT_limit
        self.use_log3 = use_log3
        self.log_noproj = log_noproj

        if self.log_noproj:
            self.log = Stiefel_Log_alg_noproj
        else:
            self.log = Stiefel_Log_alg

        if self.sampling.startswith("logitnormal"):
            _, mu, sigma = self.sampling.split("_")
            self.mu = float(mu)
            self.sigma = float(sigma)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not self.use_log3:
            return self.getitem_log4(idx)
        M = self.dataset[idx]

        ONB4_1 = M.ONB_4col.numpy()  # (N 4)
        atoms = M.atoms.squeeze(-1).numpy()
        col4 = ONB4_1[:, 3:]

        # Sample ONB_0
        ONB4_0 = sample_ONB_0(ONB4_1.shape, col4=col4)

        ONB_0 = ONB4_0[:, :3]
        ONB_1 = ONB4_1[:, :3]
        if self.use_OT:
            ONB_0 = OT_permutation_reflection(atoms, ONB_0, ONB_1, self.OT_repeats, self.OT_limit)
        # check_ONB(ONB_0, col4)

        log = self.log(ONB_0, ONB_1)
        # check_ONB_tangent(ONB_0, log)

        # Sample timestep
        if self.sampling == "linear":
            t = np.random.rand(1, 1).astype(np.float32)
        elif self.sampling == "quadratic":
            t = np.square( np.random.rand(1, 1).astype(np.float32) )
        elif self.sampling.startswith("logitnormal"):
            randn = np.random.randn(1, 1).astype(np.float32)
            t = scipy.special.expit(randn * self.sigma + self.mu)
        else:
            raise ValueError()
        t = t.clip(min=1e-5, max=(1 - 1e-5))

        # Sample ONB_t
        ONB_t = Stiefel_Exp(ONB_0, t * log)

        if self.stochastic_gamma > 0.0:
            noise = np.random.randn(*ONB_t.shape).astype(np.float32)
            noise = self.stochastic_gamma * (np.cos(np.pi*t)+1)/2 * noise
            ONB_t = Stiefel_Exp(ONB_t, project_tanspace(base=ONB_t, col4=col4, v=noise))

        # there was an extra rotate_to_mass_vector here

        target = self.log(ONB_t, ONB_1) / (1 - t)

        # check_ONB(ONB_t, col4)
        # check_ONB_tangent(ONB_t, target)

        M.t = torch.from_numpy(t)
        M.true_coords = M.coords
        M.coords = M.from_ONB_3col(torch.from_numpy(ONB_t))
        M.target = torch.from_numpy(target)

        return M
    
    def getitem_log4(self, idx):
        M = self.dataset[idx]

        ONB4_1 = M.ONB_4col.numpy()  # (N 4)
        atoms = M.atoms.squeeze(-1).numpy()
        col4 = ONB4_1[:, 3:]

        # Sample ONB_0
        ONB4_0 = sample_ONB_0(ONB4_1.shape, col4=col4)

        if self.use_OT:
            ONB3_0 = OT_permutation_reflection(atoms, ONB4_0[:, :3], ONB4_1[:, :3], self.OT_repeats, self.OT_limit)
            ONB4_0 = np.concatenate([ONB3_0, col4], axis=-1)

        log4 = self.log(ONB4_0, ONB4_1)

        # Sample timestep
        if self.sampling == "linear":
            t = np.random.rand(1, 1).astype(np.float32)
        elif self.sampling == "quadratic":
            t = np.square( np.random.rand(1, 1).astype(np.float32) )
        elif self.sampling.startswith("logitnormal"):
            randn = np.random.randn(1, 1).astype(np.float32)
            t = scipy.special.expit(randn * self.sigma + self.mu)
        else:
            raise ValueError()
        t = t.clip(min=1e-5, max=(1 - 1e-5))

        # Sample ONB_t
        ONB4_t = Stiefel_Exp(ONB4_0, t * log4)

        if self.stochastic_gamma > 0.0:
            noise = np.random.randn(ONB4_t.shape[0], 3).astype(np.float32)
            noise = self.stochastic_gamma * (np.cos(np.pi*t)+1)/2 * noise
            noise = np.concatenate([noise, np.zeros_like(col4)], axis=-1)
            A = ONB4_t.T @ noise
            tangent = noise - .5 * ONB4_t @ (A + A.T)
            ONB4_t = Stiefel_Exp(ONB4_t, tangent)
        
        ONB4_t = householder(col4) @ householder(ONB4_t[:, 3:] + col4) @ ONB4_t

        target = self.log(ONB4_t, ONB4_1) / (1 - t)

        target = target[:, :3]

        M.t = torch.from_numpy(t)
        M.true_coords = M.coords
        M.coords = M.from_ONB_4col(torch.from_numpy(ONB4_t))
        M.target = torch.from_numpy(target)

        return M



class FlowMatching(GenerativeModel):

    def __init__(
        self,
        dynamics: nn.Module,
        timesteps,
        sampling="linear",
        distance_dependence="none",
        use_OT=True,
        rejection_cutoff=100.0,
        stochastic_gamma=0.0,
        OT_repeats=5,
        OT_limit=100,
        use_log3=True,
        log_noproj=False,
    ):
        # timesteps is number of steps to take
        # for flow matching, t is a scalar between 0 and 1
        super().__init__(dynamics, timesteps)

        self.sampling = sampling
        self.distance_dependence = distance_dependence
        self.use_OT = use_OT
        self.rejection_cutoff = rejection_cutoff
        self.stochastic_gamma = stochastic_gamma
        self.OT_repeats = OT_repeats
        self.OT_limit = OT_limit
        self.use_log3 = use_log3
        self.log_noproj = log_noproj

    def wrap_dataset(self, dataset):
        return FlowMatchingDataset(
            dataset,
            sampling=self.sampling,
            distance_dependence=self.distance_dependence,
            use_OT=self.use_OT,
            rejection_cutoff=self.rejection_cutoff,
            stochastic_gamma=self.stochastic_gamma,
            OT_repeats=self.OT_repeats,
            OT_limit=self.OT_limit,
            use_log3=self.use_log3,
            log_noproj=self.log_noproj,
        )

    def objective(self, M: Molecule):
        pred = self.dynamics(M=M, moments=M.moments, t=M.t)
        delta = pred - M.target
        base = M.ONB_3col
        baseT_delta = M.scatter(base.unsqueeze(-1) * delta.unsqueeze(-2))
        norm2 = delta.square().sum() - 0.5 * baseT_delta.square().sum()
        return norm2 / M.batch_size

    @torch.no_grad()
    def sample(
        self,
        M,
        pbar=False,
        return_trajectory=False,
        inference_annealing=0.0,
        timesteps=None,
        stochastic_gamma=None,
        use_old=False,
        start=None,
        **kwargs,
    ):
        if timesteps is None:
            timesteps = self.T
        if stochastic_gamma is None:
            stochastic_gamma = self.stochastic_gamma

        M.coords = torch.zeros_like(M.coords)  # safety, so we don't cheat

        if start is None:
            coords = []
            for i in range(M.batch_size):
                mol = M[i]
                col4 = mol.masses_normalized.sqrt().cpu().numpy()
                ONB = sample_ONB_0(mol.ONB_4col.shape, col4=col4)
                ONB = torch.from_numpy(ONB).to(mol.coords)
                coords.append(mol.from_ONB_4col(ONB))
            M.coords = torch.cat(coords, dim=0)
        else:
            M.coords = start

        M_t = M
        if return_trajectory:
            trajectory = [M_t.clone().cpu()]

        ts = torch.linspace(0.0, 1.0, steps=(timesteps + 1)).tolist()
        ts_pairwise = list(zip(ts[:-1], ts[1:]))

        pbar = tqdm.tqdm(ts_pairwise, desc="Sampling", leave=False, disable=(not pbar))
        for step_id, (t, t_next) in enumerate(pbar):
            dt = t_next - t
            t = torch.full([M.batch_size, 1], t).to(M_t.moments)

            tangent = self.dynamics(M=M_t, moments=M.moments, t=t)
            if inference_annealing > 0:
                tangent = tangent * inference_annealing * (1 - t[0])
            if (stochastic_gamma > 0) and (step_id < len(ts_pairwise) - 1):
                kick = torch.randn_like(tangent)
                tangent = (tangent * dt) + (kick * np.sqrt(dt) * stochastic_gamma * (torch.cos(np.pi*t[0])+1)/2)
            else:
                tangent = (tangent * dt)

            tangent, mask = pyg.utils.to_dense_batch(tangent, M_t.batch)
            ONB_t, _ = pyg.utils.to_dense_batch(M_t.ONB_4col, M_t.batch)
            if use_old:
                ONB_t = vmapped_old_ode_step(ONB_t, tangent)
            else:
                ONB_t = vmapped_ode_step(ONB_t, tangent)
            ONB_t = ONB_t[mask]  # https://github.com/pyg-team/pytorch_geometric/discussions/6948

            M_t.coords = M_t.from_ONB_4col(ONB_t)
            if return_trajectory:
                trajectory.append(M_t.clone().cpu())

        M_t = M_t.cpu()
        if return_trajectory:
            return M_t, trajectory
        return M_t

    @torch.no_grad()
    def sample_from_formula_and_moments(
        self,
        formula,
        moments,
        batch_size=1,
        pbar=True,
        return_trajectory=False,
        device="cpu",
        dtype=torch.float32,
        **kwargs,
    ):
        # formula as str like "C_2 H_6"
        # moments as list like [260.7817, 172.0837,  15.2306]
        atoms = []
        for token in formula.split(" "):
            z, count = token.split("_")
            atoms += [int(PTABLE.GetAtomicNumber(z))] * int(count)
        atoms = torch.tensor(atoms, dtype=torch.long, device=device).unsqueeze(-1)
        moments = torch.tensor(moments, dtype=dtype, device=device).unsqueeze(0)
        n = atoms.shape[0]
        assert n > 1

        init_kwargs = dict(
            coords=torch.zeros([n, 3]).to(moments),
            atoms=atoms,
            moments=moments,
            sizes=torch.tensor([n]).to(atoms),
        )

        if batch_size > 1:
            Ms = [Molecule(**init_kwargs) for _ in range(batch_size)]
            M = pyg.data.Batch.from_data_list(Ms)
            M.batch = torch.arange(batch_size, device=device).repeat_interleave(n)
        else:
            M = Molecule(**init_kwargs)

        return self.sample(M=M, pbar=pbar, return_trajectory=return_trajectory, **kwargs)
