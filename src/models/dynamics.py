import torch
import torch.nn as nn

from src.chem import Molecule
from src.modules import AdaptiveLayerNorm, FeedForward, SignEquivariantBlock, SinusoidalEmbedding


class SignEquivariantDynamics(nn.Module):

    def __init__(
        self,
        cond_features,
        hidden_features,
        num_layers,
        act,
        num_heads=8,
        expand=4,
        **kwargs,
    ):
        super().__init__()

        ffkwargs = dict(expand=expand, act=act)

        embf = 32
        self.embed_atom = nn.Embedding(90, embf)
        self.embed_timestep = SinusoidalEmbedding(cond_features, wave_range=(1e-3, 1))
        self.embed_moments = SinusoidalEmbedding(cond_features, wave_range=(1e-4, 10000))

        self.proj_node = FeedForward(3 + embf + 1, hidden_features, **ffkwargs)
        self.proj_cond = FeedForward(4 * cond_features + 3, cond_features, expand=1, act=act, postact=act)
        self.proj_edge = FeedForward(3, hidden_features // 4, expand=1, act=act)

        self.egnn = nn.ModuleList([
            SignEquivariantBlock(
                features=hidden_features,
                edge_features=(hidden_features // 4),
                adaptive_features=cond_features,
                num_heads=num_heads,
                update_edge=(i + 1 < num_layers),
                **ffkwargs,
            )
            for i in range(num_layers)
        ])

        self.head_norm = AdaptiveLayerNorm(hidden_features, cond_features)
        self.head = FeedForward(hidden_features, 3, **ffkwargs)

    def forward(self, M: Molecule, moments, t):
        # M.check_zero_com(M.coords)

        # Node features
        pos = M.coords.abs()
        aemb = self.embed_atom(M.atoms).flatten(1)
        h = torch.cat([pos, aemb, M.masses], dim=-1)
        h = self.proj_node(h)

        # Conditioning features
        temb = self.embed_timestep(t.float()).flatten(1)  # (B C)
        memb = self.embed_moments(moments).flatten(1)  # (B 3C)
        y = torch.cat([temb, memb, moments], dim=-1)
        y = self.proj_cond(y)

        # Get base edge features
        edges = M.edge_indices()
        psrc, pdst = M.coords[edges[0]], M.coords[edges[1]]  # (B N N 3)
        dist = torch.abs(psrc - pdst)
        a = self.proj_edge(dist)

        # Pass through EGNN
        for block in self.egnn:
            h, a = block(h, cond=y, a=a, batch=M.batch_ptrs, edges=edges)
        h = self.head(self.head_norm(h, y, M.batch_ptrs))
        out = M.coords.sign() * h

        return out
