import statistics
from collections import defaultdict
from typing import Any, Dict, Literal, Optional

import lightning.pytorch as pl
import py3Dmol
import pydantic
import torch
import torch.nn as nn
import wandb
from lightning.pytorch.loggers import WandbLogger

from src.metrics import evaluate_prediction
from src.models import DDPM, FlowMatching, SignEquivariantDynamics
from src.modules.common import EMA


class LitGenConfig(pydantic.BaseModel):

    model: Literal["ddpm", "flow"] = "ddpm"

    # DDPM-specific
    ddpm_predict_eps: bool = True
    ddpm_dps: Optional[float] = None
    ddpm_project: bool = False

    # Flow-specific
    flow_sampling: str = "linear"
    flow_distance_dependence: str = "none"
    flow_use_OT: bool = True
    flow_OT_repeats: int = 5
    flow_OT_limit: int = 100
    flow_rejection_cutoff: float = 100.0
    flow_stochastic_gamma: float = 0.0
    flow_use_log3: bool = True
    flow_log_noproj: bool = False

    # ============
    # Model Fields
    # ============

    cond_features: int = 128
    hidden_features: int = 256
    num_layers: int = 6

    act: str = "silu"
    num_heads: int = 8
    expand: int = 4

    # ===============
    # Sampling Fields
    # ===============

    timesteps: int = 1000

    # ===============
    # Training Fields
    # ===============

    lr: float = 4e-4
    wd: float = 0.01
    clip_grad_norm: bool = True
    ema_decay: float = 0.999

    # ================
    # Sampling Fields
    # ================

    check_samples_every_n_epochs: int = 1
    samples_visualize_n_mols: int = 3
    samples_assess_n_batches: int = 1
    progress_bar: bool = False


class LitGen(pl.LightningModule):

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.save_hyperparameters(config)

        self.full_config = config
        self.config: LitGenConfig = LitGenConfig.parse_obj(config)
        cfg = self.config

        dynamics = SignEquivariantDynamics(**dict(config)).float()

        if cfg.model == "ddpm":
            self.model = DDPM(
                dynamics=dynamics,
                timesteps=cfg.timesteps,
                predict_eps=cfg.ddpm_predict_eps,
            )
            self.sample_kwargs = dict(dps=cfg.ddpm_dps, project=cfg.ddpm_project)
        elif cfg.model == "flow":
            self.model = FlowMatching(
                dynamics=dynamics,
                timesteps=cfg.timesteps,
                sampling=cfg.flow_sampling,
                distance_dependence=cfg.flow_distance_dependence,
                use_OT=cfg.flow_use_OT,
                rejection_cutoff=cfg.flow_rejection_cutoff,
                stochastic_gamma=cfg.flow_stochastic_gamma,
                OT_repeats=cfg.flow_OT_repeats,
                OT_limit=cfg.flow_OT_limit,
                use_log3=cfg.flow_use_log3,
                log_noproj=cfg.flow_log_noproj,
            )
            self.sample_kwargs = dict()
        else:
            raise NotImplementedError()

        self.ema = EMA(self.model, beta=cfg.ema_decay)
        grad_norm_queue = torch.full([50], fill_value=3000, dtype=torch.float)
        self.register_buffer("grad_norm_queue", grad_norm_queue)

    def wrap_dataset(self, dataset):
        return self.model.wrap_dataset(dataset)

    # Reference: https://github.com/Tony-Y/pytorch_warmup
    def linear_warmup(self, step):
        return min(step, 2000) / 2000

    def configure_optimizers(self):
        params = []
        params_no_wd = []

        for name, p in self.model.named_parameters():
            *attrs, name = name.split(".")

            # Get parent module
            parent = self.model
            for k in attrs:
                parent = getattr(parent, k)

            # Sort parameters
            if isinstance(parent, (nn.Embedding, nn.LayerNorm)) or (name == "bias"):
                params_no_wd.append(p)
            else:
                params.append(p)

        optimizer = torch.optim.AdamW(
            params=[
                {"params": params, "weight_decay": self.config.wd},
                {"params": params_no_wd, "weight_decay": 0.0},
            ],
            lr=self.config.lr,
        )

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=self.linear_warmup,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_scheduler,
                "interval": "step",
            },
        }

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if not self.config.clip_grad_norm:
            return

        max_norm = (1.5 * self.grad_norm_queue.mean()) + (2 * self.grad_norm_queue.std(unbiased=False))
        self.log("max_grad_norm", max_norm.item())

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=max_norm.item(),
            norm_type=2.0,
            error_if_nonfinite=True,
        )

        grad_norm = min(grad_norm, max_norm)
        self.grad_norm_queue[self.global_step % self.grad_norm_queue.shape[0]] = grad_norm

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model)

    def training_step(self, batch, batch_idx):
        loss = self.model.objective(batch)
        self.log("train/loss", loss, batch_size=batch.batch_size, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        old_matmul_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")  # bump up precision

        cfg = self.config
        if (
            ((self.current_epoch + 1) % cfg.check_samples_every_n_epochs == 0)
            and (batch_idx < cfg.samples_assess_n_batches)
        ):
            # Assess and visualize some samples
            num_viz = cfg.samples_visualize_n_mols if (batch_idx == 0) else 0
            self._assess_and_visualize_samples(batch, split="val", num_viz=num_viz)

        loss = self.ema.ema_model.objective(batch)
        self.log("val/loss", loss, batch_size=batch.batch_size, sync_dist=True)

        torch.set_float32_matmul_precision(old_matmul_precision)  # restore
        return loss

    @torch.no_grad()
    def _assess_and_visualize_samples(self, M, split, num_viz):
        M_preds = M.clone()
        M_trues = M.clone()
        M_trues.coords = M_trues.true_coords

        M_preds, btraj = self.ema.ema_model.sample(
            M=M_preds,
            return_trajectory=True,
            pbar=self.config.progress_bar,
            **self.sample_kwargs,
        )

        trajs = [[btraj[j][i] for j in range(len(btraj))] for i in range(num_viz)]
        M_preds = [M_preds[i].cpu() for i in range(M_preds.batch_size)]
        M_trues = [M_trues[i].cpu() for i in range(M_trues.batch_size)]

        # Compute metrics
        metrics = defaultdict(list)
        for i, (Mi_pred, Mi_true) in enumerate(zip(M_preds, M_trues)):
            sample_metrics, Mi_aln = evaluate_prediction(M_pred=Mi_pred, M_true=Mi_true)
            for k, v in sample_metrics.items():
                metrics[k].append(v)
            M_preds[i] = Mi_aln
        metrics = {f"{split}/{k}": statistics.mean(vs) for k, vs in metrics.items()}
        self.log_dict(metrics, batch_size=M.batch_size, sync_dist=True)

        # Visualize
        if (
            not isinstance(self.logger, WandbLogger)
            or (self.global_rank != 0)
            or (num_viz == 0)
        ):
            return

        grid = (num_viz, 4)
        view = py3Dmol.view(width=(200 * grid[1]), height=(200 * grid[0]), viewergrid=grid)

        for i, (Mi_pred, Mi_true, traj) in enumerate(zip(M_preds, M_trues, trajs)):
            if i >= num_viz:
                break
            view = Mi_true.show(view=view, viewer=(i, 0))
            view = Mi_pred.show(view=view, viewer=(i, 1))
            view = Mi_true.show(view=view, viewer=(i, 2))
            view = Mi_pred.show(view=view, viewer=(i, 2), color="cyan")
            view = show_traj(traj, view=view, viewer=(i, 3))

        view.render()
        t = view.js()
        js = t.startjs + t.endjs
        wandb.log({f"{split}_samples": wandb.Html(js), "epoch": self.current_epoch})


def show_traj(traj, view, viewer):
    assert isinstance(traj, list)

    trajfile = ""
    for M in traj:
        trajfile += M.xyzfile
    for _ in range(10):
        trajfile += traj[-1].xyzfile

    view.addModelsAsFrames(trajfile, "xyz", viewer=viewer)
    view.setStyle({"model": -1}, {"stick": {"radius": 0.2}, "sphere": {"scale": 0.2}}, viewer=viewer)
    view.animate({"interval": 10})
    view.zoomTo()
    return view
