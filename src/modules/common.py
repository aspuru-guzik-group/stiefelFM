import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Activation(nn.Module):
    #
    ACTIVATION_REGISTRY = {
        "silu": F.silu,
        "gelu": F.gelu,
        "relu": F.relu,
        "none": lambda z: z,
    }

    def __init__(self, name, glu=False):
        super().__init__()

        self.fn = self.ACTIVATION_REGISTRY[name]
        self.glu = glu

    def forward(self, input):
        if self.glu:
            gate, input = input.chunk(2, dim=-1)
            return input * self.fn(gate)
        else:
            return self.fn(input)


class AdaptiveLayerNorm(nn.Module):

    def __init__(self, features, adaptive_features):
        super().__init__()

        self.norm = nn.LayerNorm(features, elementwise_affine=False)
        self.proj = nn.Linear(adaptive_features, 2 * features)

    def forward(self, input, cond, batch):
        params = self.proj(cond)
        scale, shift = params.chunk(chunks=2, dim=-1)
        scale = 1 + scale

        scale = scale.index_select(0, batch)
        shift = shift.index_select(0, batch)
        return torch.addcmul(shift, self.norm(input), scale)


class SinusoidalEmbedding(nn.Module):

    def __init__(self, embedding_dim, wave_range):
        assert embedding_dim % 2 == 0
        super().__init__()

        half_dim = embedding_dim // 2
        wavelengths = np.geomspace(*wave_range, num=half_dim)
        freqs = 2 * torch.pi / torch.from_numpy(wavelengths).float()
        self.register_buffer("freqs", freqs)

    def forward(self, input):
        temb = input.unsqueeze(-1) * self.freqs
        temb = torch.cat([temb.sin(), temb.cos()], dim=-1)
        return temb


class FeedForward(nn.Module):

    def __init__(
        self,
        features, out_features=None,
        expand=1,
        act="silu", postact="none", glu=False,
    ):
        super().__init__()

        if out_features is None:
            out_features = features
        width = max(features, out_features)

        if glu:
            inner_features = int(width * expand * (2 / 3))
            inner_features = [2 * inner_features, inner_features]
        else:
            inner_features = [width * expand] * 2

        self.mlp = nn.Sequential(
            nn.Linear(features, inner_features[0]),
            Activation(act, glu=glu),
            nn.Linear(inner_features[1], out_features),
            Activation(postact),
        )

    def forward(self, input):
        return self.mlp(input)


class EMA(nn.Module):

    def __init__(self, model, beta=0.999):
        super().__init__()

        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        self.beta = beta

    def update(self, model):
        for p, p_ema in zip(model.parameters(), self.ema_model.parameters()):
            p_ema.data.lerp_(p, weight=(1.0 - self.beta))
