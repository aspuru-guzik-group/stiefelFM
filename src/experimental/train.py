import datetime
from typing import List, Literal, Optional

import lightning.pytorch as pl
import pydantic_cli
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger

from src.datamodule import ConformerDataModule
from src.experimental.callbacks import GradNormCallback
from src.experimental.lit import LitGen, LitGenConfig
from src.paths import LOG_DIR, random_checkpoint_dir
import os

class TrainGenConfig(LitGenConfig):

    seed: int = 100
    debug: bool = False
    debug_nan: bool = False

    accelerator: str = "cpu"
    num_nodes: int = 1
    devices: int = 1
    strategy: Optional[str] = "auto"

    matmul_precision: Literal["medium", "high", "highest"] = "high"

    # =================
    # Datamodule Fields
    # =================

    dataset: Literal["qm9", "geom"] = "qm9"

    batch_size_train: int = 512
    batch_size_eval: int = 1024
    split_ratio: List[float] = (0.8, 0.1, 0.1)
    num_workers: int = 8

    # ===============
    # Training Fields
    # ===============

    max_epochs: int = 1500

    resume_path: Optional[str] = None

    # ==============
    # Logging Fields
    # ==============

    wandb: bool = False
    wandb_project: str = "train_kreed_moments"
    wandb_entity: Optional[str] = "matter-lab"
    wandb_run_name: Optional[str] = None

    checkpoint: bool = False
    checkpoint_dir: Optional[str] = None
    checkpoint_every_n_minutes: int = 10 # unused, but kept for compatibility

    log_every_n_steps: int = 10

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def train(config: TrainGenConfig):
    cfg = config

    if cfg.debug_nan:
        torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Seeding
    pl.seed_everything(cfg.seed, workers=True)

    # Initialize and load model
    model = LitGen(config=dict(cfg))

    # Load dataset
    data = ConformerDataModule(
        dataset=cfg.dataset,
        batch_size_train=cfg.batch_size_train,
        batch_size_eval=cfg.batch_size_eval,
        num_workers=cfg.num_workers,
        wrapper=model.wrap_dataset,
    )

    # Initialize trainer
    callbacks = [
        ModelSummary(max_depth=2),
        GradNormCallback(),
    ]

    if cfg.wandb:
        project = cfg.wandb_project + ("_debug" if cfg.debug else "")
        logger = WandbLogger(
            name=cfg.wandb_run_name,
            project=project,
            entity=cfg.wandb_entity,
            log_model=False,
            save_dir=LOG_DIR,
        )
        callbacks.append(LearningRateMonitor())
    else:
        logger = False

    if cfg.checkpoint:
        if cfg.checkpoint_dir is None:  # set to some random unique folder
            cfg.checkpoint_dir = random_checkpoint_dir()
        callbacks.extend([
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                filename="epoch={epoch}-step={step}-validity={val/validity:.3f}",
                monitor="val/validity",
                mode="max",
                save_top_k=2,
                every_n_epochs=cfg.check_samples_every_n_epochs,
                verbose=True,
                auto_insert_metric_name=False,
            ),
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                monitor="epoch",
                mode="max",
                save_top_k=2,
                # Every 10 minutes, commented out because it's not reproducible
                # train_time_interval=datetime.timedelta(minutes=cfg.checkpoint_every_n_minutes),
                every_n_epochs=1,  # for reliability
                save_last=True,
            ),
        ])

    if cfg.debug:
        debug_kwargs = {"limit_train_batches": 10, "limit_val_batches": 10}
    else:
        debug_kwargs = {}

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        num_nodes=cfg.num_nodes,
        devices=cfg.devices,
        strategy=cfg.strategy,
        callbacks=callbacks,
        enable_checkpointing=cfg.checkpoint,
        logger=logger,
        max_epochs=cfg.max_epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        enable_progress_bar=cfg.progress_bar,
        use_distributed_sampler=True,
        **debug_kwargs,
    )

    # Start training
    resume_path = cfg.resume_path
    if resume_path is not None and not os.path.exists(resume_path):
        # ignore resume_path if it does not exist
        print(f"Warning: {resume_path} does not exist. Ignoring it.")
        resume_path = None
    trainer.fit(model=model, datamodule=data, ckpt_path=resume_path)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TrainGenConfig, train)
