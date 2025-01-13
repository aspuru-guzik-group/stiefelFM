import numpy as np
import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import tqdm

from src.chem import Molecule
from src.models.base import GenerativeModel


def clip_noise_schedule(alphas2, margin=0.001):
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = (alphas2[1:] / alphas2[:-1])
    alphas_step = np.clip(alphas_step, a_min=margin, a_max=1.0)
    return np.cumprod(alphas_step, axis=0)


def polynomial_schedule(timesteps, s=1e-5, power=2.0):
    T = timesteps
    t = np.linspace(0, T, T + 1)
    alphas2 = (1 - np.power(t / T, power)) ** 2
    alphas2 = clip_noise_schedule(alphas2, margin=0.001)
    return (1 - 2 * s) * alphas2 + s


class NoiseSchedule(nn.Module):

    def __init__(self, timesteps):
        super().__init__()

        self.timesteps = timesteps

        alphas2 = polynomial_schedule(timesteps, s=1e-5, power=2.0)
        alphas2 = torch.from_numpy(alphas2)
        gammas = torch.log(1.0 - alphas2) - torch.log(alphas2)
        self.register_buffer("gammas", gammas.float())

    def forward(self, t):
        assert not torch.is_floating_point(t)
        return self.gammas[t.long()]


def gamma_to_sigma(gamma):
    return torch.sigmoid(gamma).sqrt()


def gamma_to_alpha(gamma):
    return torch.sigmoid(-gamma).sqrt()


def gamma_to_SNR(gamma):
    return torch.exp(-gamma)


def sample_M_randn_like(M, mean=None, std=None, return_noise=False):
    eps = torch.randn_like(M.coords)
    coords = eps if (mean is None) else (mean + std * eps)
    coords = M.center_com(coords, orthogonal=True)

    if not torch.isfinite(coords).all():
        print("(!!!) NaNs detected, setting to 0.")
        coords = torch.zeros_like(coords)

    M = M.clone()
    M.coords = coords
    return (M, eps) if return_noise else M


class DDPMDataset(pyg.data.InMemoryDataset):

    def __init__(self, dataset, timesteps, predict_eps):
        super().__init__()

        self.dataset = dataset
        self.T = timesteps
        self.predict_eps = predict_eps

        self.gamma = NoiseSchedule(timesteps)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        M = self.dataset[idx]

        t = torch.randint(0, self.T + 1, size=[1, 1], device=M.device)
        gamma_t = M.broadcast(self.gamma(t))
        alpha_t = gamma_to_alpha(gamma_t)
        sigma_t = gamma_to_sigma(gamma_t)

        Mt, eps = sample_M_randn_like(
            M=M,
            mean=(alpha_t * M.coords),
            std=sigma_t,
            return_noise=True,
        )
        Mt.t = t
        Mt.true_coords = M.coords
        Mt.target = eps if self.predict_eps else M.coords

        return Mt


class DDPM(GenerativeModel):

    def __init__(self, dynamics: nn.Module, timesteps, predict_eps):
        super().__init__(dynamics, timesteps)

        self.predict_eps = predict_eps
        self.gamma = NoiseSchedule(timesteps)

    def wrap_dataset(self, dataset):
        return DDPMDataset(dataset, timesteps=self.T, predict_eps=self.predict_eps)

    def forward(self, M: Molecule, t):
        out = self.dynamics(M=M, moments=M.moments, t=(t.float() / self.T))
        return M.center_com(out, orthogonal=False)

    def objective(self, M: Molecule):
        return F.mse_loss(self(M=M, t=M.t), M.target)

    def denoise(self, M_t, t):
        if self.predict_eps:
            gamma_t = M_t.broadcast(self.gamma(t))
            sigma_t = gamma_to_sigma(gamma_t)
            alpha_t = gamma_to_alpha(gamma_t)
            return 1.0 / alpha_t * (M_t.coords - sigma_t * self(M=M_t, t=t))
        else:
            return self(M=M_t, t=t)

    def sigma_and_alpha_t_given_s(self, gamma_t, gamma_s):
        sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def sample_Ms_given_Mt(self, M_t, s, t, dps=None):
        assert torch.all(s < t)
        gamma_s = M_t.broadcast(self.gamma(s))
        gamma_t = M_t.broadcast(self.gamma(t))

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)

        alpha_s = gamma_to_alpha(gamma_s)
        sigma_s = gamma_to_sigma(gamma_s)
        sigma_t = gamma_to_sigma(gamma_t)

        if dps is not None:
            torch.set_grad_enabled(True)
            M_t.coords.detach_()
            M_t.coords.requires_grad = True

        out = self.denoise(M_t=M_t, t=t)

        with torch.no_grad():  # wrapped in case DPS is on
            mu = (alpha_t_given_s * sigma_s.square() * M_t.coords) + (alpha_s * sigma2_t_given_s * out)
            mu = mu / sigma_t.square()
            sigma = sigma_t_given_s * sigma_s / sigma_t
            M_s = sample_M_randn_like(M_t, mean=mu, std=sigma)

        if dps is not None:
            m = M_t.masses
            moments_error = M_t.scatter(m * out.square()) - M_t.moments  # (B 3)
            offdiag_error = M_t.scatter(m * out * out.roll(shifts=1, dims=-1))  # (B 3)
            error = torch.cat([moments_error, offdiag_error], dim=-1)  # (B 6)
            error = error / M_t.scatter(m)  # normalize to same scale
            norm = LA.vector_norm(error, dim=-1)
            norm_grad = torch.autograd.grad(outputs=norm.sum(), inputs=M_t.coords)[0]
            norm_grad = M_t.center_com(norm_grad, orthogonal=True).detach()
            M_s.coords -= dps * norm_grad
            torch.set_grad_enabled(False)

        return M_s

    def sample_M_given_M0(self, M_0):
        zeros = torch.zeros([M_0.batch_size, 1], dtype=torch.int, device=M_0.device)
        gamma_0 = M_0.broadcast(self.gamma(zeros))

        mu = self.denoise(M_t=M_0, t=zeros)
        sigma = gamma_to_SNR(-0.5 * gamma_0)
        return sample_M_randn_like(M_0, mean=mu, std=sigma)

    @torch.no_grad()
    def sample(self, M, pbar=False, return_trajectory=False, dps=None, project=False, **kwargs):
        M_T = sample_M_randn_like(M)  # M_T
        traj = [M_T.clone().cpu()] if return_trajectory else None
        del M  # safety

        M_t = M_T
        countdown = list(reversed(range(0, self.T)))
        for step in tqdm.tqdm(countdown, desc="Sampling", leave=False, disable=(not pbar)):
            s = torch.full(size=[M_t.batch_size, 1], fill_value=step, device=M_t.device)
            M_t = self.sample_Ms_given_Mt(M_t=M_t, s=s, t=(s + 1), dps=dps)
            if return_trajectory:
                traj.append(M_t.clone().cpu())

        M = self.sample_M_given_M0(M_t)
        M.check_zero_com(M.coords)

        if project:
            from src.models.flow import householder
            coords = []
            for i in range(M.batch_size):
                mol = M[i]
                ONB = mol.ONB_4col
                col4 = ONB[:, 3:]
                U, _, V = torch.linalg.svd(ONB)
                ONB = U[:, :4] @ V
                ONB = householder(col4, usetorch=True) @ householder(ONB[:, 3:] + col4, usetorch=True) @ ONB
                coords.append(mol.from_ONB_4col(ONB))
            M.coords = torch.cat(coords, dim=0)

        M = M.cpu()
        return (M, traj) if return_trajectory else M
