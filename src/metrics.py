import itertools
import os
import subprocess
import uuid

import einops
import numpy as np
import scipy
import scipy.optimize  # needed for linear_sum_assignment
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger

from src.chem import Molecule

folder = uuid.uuid4().hex
folder = f"/tmp/{folder}"
os.makedirs(folder, exist_ok=True)


# =========================================================================== #
#                                   Caches                                    #
# =========================================================================== #


def _build_transforms_cache():
    transforms = []
    for flips in itertools.product([-1, 1], repeat=3):
        transforms.append(torch.diag_embed(torch.tensor(flips, dtype=torch.float)))
    return torch.stack(transforms, dim=0)


TRANSFORMS = _build_transforms_cache()


# =========================================================================== #
#                                   Metrics                                   #
# =========================================================================== #


def compute_coord_rmse(atoms_pred, atoms_true, coords_pred, coords_true):
    transformed_coords_preds = einops.einsum(
        TRANSFORMS.to(coords_pred),
        coords_pred,
        "t i j, n j -> t n i"
    ).unsqueeze(-2)  # (T N 1 3)

    # An T x N x N matrix where transformed_costs[t][i][j] is the cost of assigning atom #i in
    # coords_pred (under the t-th transformation) to atom #j in coords_true.
    # For our purposes, the cost between atoms i and j is their squared distance.
    # However, we have to be careful about not assigning two atoms of different types together.
    # To avoid this, we can set their cost to infinity.
    transformed_costs = torch.square(transformed_coords_preds - coords_true).sum(dim=-1)
    transformed_costs = torch.where(atoms_pred == atoms_true.T, transformed_costs, torch.inf)
    transformed_costs = transformed_costs.cpu().numpy()

    # RMSD = root mean squared distance, but call it rmse
    rmses = []
    for cost in transformed_costs:
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
        rmses.append(np.sqrt(np.mean(cost[row_ind, col_ind])))
    rmses = torch.tensor(rmses).to(coords_pred)

    idx = rmses.argmin()
    return rmses[idx].item(), TRANSFORMS[idx]


def compute_correctness(M_pred, M_true):
    try:
        smiles_pred = M_pred.smiles
        smiles_true = M_true.smiles
        return float(smiles_pred == smiles_true)
    except:
        return 0.0


# def compute_stability(M_pred, M_true):
#     try:
#         mol = Chem.MolFromXYZBlock(M_pred.xyzfile)
#         rdDetermineBonds.DetermineBonds(mol)
#         ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol)
#         pred_force_norm = np.linalg.norm(ff.CalcGrad())
#         # pred_energy = ff.CalcEnergy()
#         pred_is_valid = True
#     except:
#         pred_is_valid = False
#     try:
#         mol = Chem.MolFromXYZBlock(M_true.xyzfile)
#         rdDetermineBonds.DetermineBonds(mol)
#         ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol)
#         true_force_norm = np.linalg.norm(ff.CalcGrad())
#         # true_energy = ff.CalcEnergy()
#         true_is_valid = True
#     except:
#         true_is_valid = False

#     # if true is not valid, return 1.0 stable
#     # if pred is not valid, return 0.0 stable
#     # if both valid, return pred_force_norm < 1.2 * true_force_norm
#     stable = (not true_is_valid) or (pred_is_valid and pred_force_norm < 1.2 * true_force_norm)
#     return float(stable)


def xtb_single_point(M):
    # Write the molecule to an XYZ file
    file = f"{folder}/xtb_input.xyz"
    with open(file, "w") as f:
        f.write(M.xyzfile)

    # Define the command to run xTB with the input file
    command = ["xtb", file]

    try:
        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=folder)
        # Parse the output to find the energy
        output_lines = result.stdout.split("\n")
        for i in range(len(output_lines)):
            if "TOTAL ENERGY" in output_lines[i]:
                energy_line = output_lines[i]
                gradient_norm_line = output_lines[i + 1]
                break
        gradient_norm = float(gradient_norm_line.split()[-3])
        energy = float(energy_line.split()[-3])
        return gradient_norm, energy
    except subprocess.CalledProcessError as e:
        # If xTB returns a non-zero exit code, print the error message
        return 1.0, 1.0


def compute_validity(M_pred, M_true):
    RDLogger.DisableLog("rdApp.*")
    try:
        smi = M_pred.smiles
        pred_is_valid = ("." not in smi) and (Chem.MolFromSmiles(smi, sanitize=True) is not None)
    except Exception:
        pred_is_valid = False
    try:
        smi = M_true.smiles
        true_is_valid = ("." not in smi) and (Chem.MolFromSmiles(smi, sanitize=True) is not None)
    except Exception:
        true_is_valid = False
    RDLogger.DisableLog("rdApp.*")
    return float((not true_is_valid) or pred_is_valid)


@torch.no_grad()
def evaluate_prediction(M_pred: Molecule, M_true: Molecule, return_aligned=True, singlepoint=True):
    # don't diagonalize true coords: if it flips axes, then we are aligning to 
    # the wrong reflection of the ground truth
    coords_true = M_true.true_coords
    moments_true = M_true.moments

    coords_pred, moments_pred = M_pred.rotated_to_principal_axes(return_moments=True)

    atoms_true = M_true.atoms  # (N 1)
    atoms_pred = M_pred.atoms  # (N 1)

    # Planar dyadic deviation
    moments_rmse = F.mse_loss(
        torch.triu(M_pred.planar_dyadic.squeeze(0)),
        torch.triu(torch.diag(moments_true.squeeze(0)))
    ).sqrt().item()
    # scaled_moments_rmse = moments_rmse / M_true.masses.sum().item()

    # Validity
    validity = compute_validity(M_pred=M_pred, M_true=M_true)

    # Stability
    if singlepoint:
        grad_norm, energy = xtb_single_point(M_pred)
    else:
        grad_norm, energy = 1.0, 1.0

    # Correctness and RMSE
    correctness = compute_correctness(M_pred=M_pred, M_true=M_true)
    coord_rmse, _ = compute_coord_rmse(
        atoms_pred=atoms_pred,
        atoms_true=atoms_true,
        coords_pred=coords_pred,
        coords_true=coords_true,
    )

    # Correctness and RMSE on aligned heavy atom coordinates
    heavy_mask_pred = (atoms_pred.squeeze(-1) != 1)
    heavy_atoms_pred = atoms_pred[heavy_mask_pred]
    heavy_mask_true = (atoms_true.squeeze(-1) != 1)
    heavy_atoms_true = atoms_true[heavy_mask_true]
    heavy_M_pred = M_pred.clone()
    heavy_M_pred.atoms = heavy_atoms_pred
    heavy_M_pred.coords = coords_pred[heavy_mask_pred]
    heavy_M_true = M_true.clone()
    heavy_M_true.atoms = heavy_atoms_true
    heavy_M_true.coords = coords_true[heavy_mask_true]

    heavy_correctness = compute_correctness(M_pred=heavy_M_pred, M_true=heavy_M_true)
    heavy_coord_rmse, transform = compute_coord_rmse(
        atoms_pred=heavy_M_pred.atoms,
        atoms_true=heavy_M_true.atoms,
        coords_pred=heavy_M_pred.coords,
        coords_true=heavy_M_true.coords,
    )

    rmsd_under_pt10 = float(coord_rmse < 0.1)
    rmsd_under_pt25 = float(coord_rmse < 0.25)
    log_grad_norm = np.log10(grad_norm)

    metrics = {
        "moments_rmse": moments_rmse,
        # "scaled_moments_rmse": scaled_moments_rmse,
        "validity": validity,
        "correctness": correctness,
        "heavy_correctness": heavy_correctness,
        "coord_rmse": coord_rmse,
        "heavy_coord_rmse": heavy_coord_rmse,
        "grad_norm": grad_norm,
        "log_grad_norm": log_grad_norm,
        "energy": energy,
        "rmsd_under_pt10": rmsd_under_pt10,
        "rmsd_under_pt25": rmsd_under_pt25,
    }

    coords_aln = einops.einsum(transform, coords_pred, "i j, n j -> n i")
    M_aln = M_pred.clone()
    M_aln.coords = coords_aln
    return (metrics, M_aln) if return_aligned else metrics


from src.stiefel_log import Stiefel_Log_alg, Stiefel_Exp, metric
def compute_curve_length(M, coords_traj):
    ONB_traj = []
    for i in range(len(coords_traj)):
        Mt = M.clone()
        Mt.coords = coords_traj[i]
        ONB_traj.append(Mt.ONB_4col.numpy())
    
    length = 0
    for i in range(len(ONB_traj)-1):
        Delta = Stiefel_Log_alg(ONB_traj[i], ONB_traj[i+1])
        length += np.sqrt(metric(ONB_traj[i], Delta, Delta))
    
    return length


def compute_diversity(atoms, coords_list):
        
    # average pairwise RMSD
    sum_rmsd = 0
    count = 0
    n = len(coords_list)
    for i in range(n):
        for j in range(i+1, n):
            sum_rmsd += compute_coord_rmse(atoms, atoms, coords_list[i], coords_list[j])[0]
            count += 1
    return sum_rmsd / count
