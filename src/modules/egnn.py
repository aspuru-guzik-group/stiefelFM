import einops
import torch.nn as nn
import torch_geometric as pyg

from src.modules.common import Activation, AdaptiveLayerNorm, FeedForward


class SignEquivariantBlock(nn.Module):

    def __init__(
        self,
        features,
        edge_features,
        adaptive_features,
        act,
        num_heads=8,
        update_edge=True,
        expand=4,
    ):
        assert features % num_heads == 0
        super().__init__()

        self.features = features
        self.num_heads = num_heads
        self.head_dim = features // num_heads
        self.update_edge = update_edge

        # Broken down like this for efficiency
        self.norm_mix = AdaptiveLayerNorm(features, adaptive_features)
        self.proj_h = nn.Linear(features, 2 * features)
        self.proj_a = nn.Linear(edge_features, features, bias=False)
        self.act = Activation(act)
        self.proj_vatt = nn.Linear(features, features + num_heads, bias=False)
        self.proj_out = nn.Linear(features, features)

        self.norm_edge = nn.LayerNorm(edge_features)
        if update_edge:
            self.proj_edge = nn.Linear(features, edge_features)

        self.norm_mlp = AdaptiveLayerNorm(features, adaptive_features)
        self.mlp = FeedForward(features, expand=expand, act=act)

    def forward(self, input, cond, a, batch, edges):
        N = input.shape[0]

        # Compute messages
        h = input
        hsrc, hdst = self.proj_h(self.norm_mix(h, cond, batch)).chunk(2, dim=-1)
        hsrc, hdst = hsrc[edges[0]], hdst[edges[1]]
        m = self.act(hsrc + hdst + self.proj_a(self.norm_edge(a)))

        v, w = self.proj_vatt(m).split([self.features, self.num_heads], dim=-1)
        if self.update_edge:
            a = a + self.proj_edge(m)

        # Update nodes
        w = pyg.utils.softmax(w, index=edges[0], num_nodes=N)
        w = einops.repeat(w, "e h -> e (h d)", d=self.head_dim)
        o = pyg.nn.global_add_pool(w * v, edges[0], size=N)
        h = h + self.proj_out(o)

        # MLP
        h = h + self.mlp(self.norm_mlp(h, cond, batch))

        return h, a
