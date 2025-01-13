import statistics
from collections import defaultdict
from typing import List, Literal, Optional

import lightning.pytorch as pl
import pydantic
import pydantic_cli
from lightning.pytorch.loggers import WandbLogger

from src.datamodule import ConformerDataModule
from src.experimental.lit import LitGen
from src.metrics import evaluate_prediction
from src.paths import LOG_DIR, REPO_ROOT


class TestDDPMConfig(pydantic.BaseModel):
    #
    checkpoint_path: str = str(REPO_ROOT / "checkpoints" / "last.ckpt")

    seed: int = 100
    debug: bool = False

    accelerator: str = "cpu"
    num_nodes: int = 1
    devices: int = 1
    strategy: Optional[str] = "auto"

    # =================
    # Datamodule Fields
    # =================

    dataset: Literal["qm9", "geom"] = "qm9"

    batch_size: int = 512
    split_ratio: List[float] = (0.8, 0.1, 0.1)
    num_workers: int = 0

    # ===============
    # Sampling Fields
    # ===============

    dps: Optional[float] = None

    # ==============
    # Logging Fields
    # ==============

    wandb_project: str = "test_kreed_moments"
    wandb_entity: Optional[str] = "matter-lab"

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


class LitTestDDPM(pl.LightningModule):

    def __init__(self, ddpm, dps):
        super().__init__()

        self.ddpm = ddpm
        self.ddpm.eval()

        self.dps = dps

    def test_step(self, batch):
        M, moments = batch

        M_trues = M
        M_preds = self.ddpm.sample(M=M, moments=moments, dps=self.dps)
        exit()

        # Unbatch
        M_trues = M_trues.cpu().unbatch()
        M_preds = M_preds.cpu().unbatch()

        # Compute metrics
        metrics = defaultdict(list)
        for i, (Mi_pred, Mi_true) in enumerate(zip(M_preds, M_trues)):
            sample_metrics, Mi_aln = evaluate_prediction(M_pred=Mi_pred, M_true=Mi_true)
            for k, v in sample_metrics.items():
                metrics[k].append(v)
            M_preds[i] = Mi_aln
        metrics = {f"test/{k}": statistics.mean(vs) for k, vs in metrics.items()}

        self.log_dict(metrics, batch_size=M.batch_size, sync_dist=True)


def test(config: TestDDPMConfig):
    raise NotImplementedError()  # FIXME: this is broken after refactoring
    cfg = config

    # Seeding
    pl.seed_everything(cfg.seed, workers=True)

    # Load dataset
    data = ConformerDataModule(
        dataset=cfg.dataset,
        seed=cfg.seed,
        batch_size_train=None,  # not used
        batch_size_eval=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Initialize and load DDPM
    model = LitGen.load_from_checkpoint(cfg.checkpoint_path, map_location="cpu")
    model = LitTestDDPM(model.ddpm, dps=cfg.dps)

    # Initialize trainer
    logger = WandbLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        log_model=False,
        save_dir=LOG_DIR,
    )

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        num_nodes=cfg.num_nodes,
        devices=cfg.devices,
        strategy=cfg.strategy,
        enable_checkpointing=False,
        logger=logger,
        use_distributed_sampler=True,
        inference_mode=False,  # important for DPS
    )

    # Start testing
    trainer.test(model=model, datamodule=data)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TestDDPMConfig, test)
