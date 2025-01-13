import torch
import torch.linalg as LA
import torch_geometric as pyg
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import GetPeriodicTable


# =========================================================================== #
#                                   Caches                                    #
# =========================================================================== #


def _build_edge_cache(max_nodes):
    cache = []
    for i in range(max_nodes):
        for j in range(i):
            cache.append([i, j])
            cache.append([j, i])
    return torch.tensor(cache, dtype=torch.long).mT


_EDGE_CACHE = _build_edge_cache(max_nodes=200)

PTABLE = GetPeriodicTable()
ATOM_MASSES = [0] + [PTABLE.GetMostCommonIsotopeMass(z) for z in range(1, 119)]
ATOM_MASSES = torch.tensor(ATOM_MASSES, dtype=torch.float32)


# =========================================================================== #
#                                  Molecule                                   #
# =========================================================================== #


class Molecule(pyg.data.Data):

    def broadcast(self, input):
        return torch.repeat_interleave(input, self.sizes, dim=0)

    def scatter(self, input, reduce="sum", repeat=False):
        out = pyg.utils.scatter(
            input,
            index=self.batch_ptrs,
            dim=0, dim_size=self.batch_size,
            reduce=reduce,
        )
        if repeat:
            out = self.broadcast(out)
        return out

    def edge_indices(self):
        E, offset = [], 0
        for n in self.sizes:
            E.append(offset + _EDGE_CACHE[:, :(n * (n - 1))].to(self.device))
            offset += n
        return torch.cat(E, dim=-1)

    # ==============
    # Chem Utilities
    # ==============

    @property
    def xyzfile(self):
        assert not self.batched
        file = f"{self.coords.shape[0]}\n\n"
        for a, p in zip(self.atoms, self.coords):
            x, y, z = p.tolist()
            file += f"{PTABLE.GetElementSymbol(int(a))} {x:f} {y:f} {z:f}\n"
        return file

    def show(self, view=None, viewer=None, color=None):
        assert not self.batched
        if view is None:
            import py3Dmol

            view = py3Dmol.view(width=400, height=400)
        view.addModel(self.xyzfile, "xyz", viewer=viewer)
        style = {"stick": {"radius": 0.2}, "sphere": {"scale": 0.2}}
        if color is not None:
            style["stick"]["color"] = color
            style["sphere"]["color"] = color
        view.setStyle({"model": -1}, style, viewer=viewer)
        # view.zoomTo()
        return view

    @property
    def smiles(self):
        assert not self.batched
        mol = Chem.MolFromXYZBlock(self.xyzfile)
        rdDetermineBonds.DetermineConnectivity(mol)
        return Chem.MolToSmiles(mol)

    @property
    def formula(self):
        assert not self.batched
        formula_dict = dict()
        for a in self.atoms:
            symbol = PTABLE.GetElementSymbol(int(a))
            formula_dict[symbol] = formula_dict.get(symbol, 0) + 1
        return " ".join([f"{k}_{v}" for k, v in formula_dict.items()])

    # ==========
    # Properties
    # ==========

    @property
    def batch_size(self):
        return self.sizes.numel()

    @property
    def batched(self):
        return self.batch_size > 1

    @property
    def batch_ptrs(self):
        ptrs = torch.arange(self.batch_size).to(self.sizes)
        return torch.repeat_interleave(ptrs, self.sizes, dim=0)

    @property
    def device(self):
        return self.coords.device

    @property
    def masses(self):
        return ATOM_MASSES.to(self.device)[self.atoms]

    @property
    def masses_normalized(self):
        return self.masses / self.scatter(self.masses, repeat=True)

    @property
    def ONB_3col(self):
        return self.coords * (self.masses / self.broadcast(self.moments)).sqrt()

    @property
    def ONB_4col(self):
        return torch.cat([self.ONB_3col, self.masses_normalized.sqrt()], dim=-1)
    
    def from_ONB_3col(self, ONB):
        return ONB * (self.broadcast(self.moments) / self.masses).sqrt()

    def from_ONB_4col(self, ONB):
        return ONB[..., :3] * (self.broadcast(self.moments) / self.masses).sqrt()

    def center_com(self, coords, orthogonal=False):
        m = self.masses_normalized
        coms = self.scatter(m * coords, repeat=True)
        if orthogonal:
            norm2 = self.scatter(m.square(), repeat=True)
            shifts = m * coms / norm2
        else:
            shifts = coms
        return coords - shifts

    @torch.no_grad()
    def check_zero_com(self, coords):
        coms = self.scatter(self.masses_normalized * coords)
        drift = coms.abs().max().item()
        if drift > 1e-4:
            print(f"WARNING: CoM off origin by {drift =}")
            return False
        return True

    @property
    def planar_dyadic(self):
        coords = self.center_com(self.coords)
        m = self.masses

        dyadic = m.unsqueeze(-1) * coords.unsqueeze(-1) * coords.unsqueeze(-2)
        dyadic = self.scatter(dyadic)
        return dyadic

    def rotated_to_principal_axes(self, return_moments=False):
        coords, m = self.coords.double(), self.masses.double()  # (N 3) (N 1)

        # Subtract CoM
        coords = self.center_com(coords)

        # Compute planar dyadic
        # (N 1 1) * (N 3 1) * (N 1 3)
        dyadic = m.unsqueeze(-1) * coords.unsqueeze(-1) * coords.unsqueeze(-2)
        dyadic = self.scatter(dyadic)  # (B 3 3)

        # Diagonalize
        moments, V = LA.eigh(dyadic)  # (B 3) (B 3 3)

        # Sort eigenvalues in descending order
        moments = torch.flip(moments, dims=[-1])
        V = torch.flip(V, dims=[-1])

        # Sanity check
        Q = V @ torch.diag_embed(moments) @ V.mT
        error = (dyadic - Q).abs().max().item()
        if error > 1e-5:
            print(f"WARNING: instability in diagonalizing planar dyadic ({error = })")

        coords = torch.einsum("n j, n j i -> n i", coords, self.broadcast(V))
        coords = coords.float()
        return (coords, moments.float()) if return_moments else coords

    def project_to_manifold(self):
        assert not self.batched

        from src.models.flow import householder
        ONB = self.ONB_4col
        p = 4

        col4 = ONB[:, 3:]
        # project to manifold
        U, _, V = torch.linalg.svd(ONB)
        ONB = U[..., :, :p] @ V

        # rotate to col4
        ONB = householder(col4, usetorch=True) @ householder(ONB[:, 3:] + col4, usetorch=True) @ ONB
        return self.from_ONB_4col(ONB)
