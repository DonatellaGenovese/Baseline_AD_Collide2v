from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAUROC

from .components.autoencoder import AutoEncoder


class AELitModule(LightningModule):
    """LightningModule for unsupervised anomaly detection with an Autoencoder.

    Trains on background-only events (reconstruction loss).
    Anomaly score = per-event MSE reconstruction error.
    AUROC is computed on val/test sets that contain both background and signal.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        lr: float,
        weight_decay: float,
        optimizer: Callable,
        scheduler: Optional[Callable],
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = None  # built in setup() once vlen is known from datamodule

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.test_auroc = BinaryAUROC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-event reconstruction MSE — higher means more anomalous."""
        x_hat = self.forward(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

    def on_train_start(self) -> None:
        self.val_loss.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch  # labels not used during background-only training
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        val_loss = self.val_loss.compute()
        import sys
        sys.__stdout__.write(f"\n[Epoch {self.current_epoch:03d}] val/loss={val_loss:.4f}\n")
        sys.__stdout__.flush()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        scores = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)
        self.test_auroc.update(scores, y)

    def on_test_epoch_end(self) -> None:
        try:
            auroc = self.test_auroc.compute()
        except Exception:
            auroc = torch.tensor(0.0, device=self.device)
        self.test_auroc.reset()
        self.log("test/auroc", auroc, prog_bar=True)

    def setup(self, stage: str) -> None:
        if self.trainer and getattr(self.trainer, "datamodule", None):
            vlen = getattr(self.trainer.datamodule, "vlen", None)
            self.net = AutoEncoder(
                in_dim=vlen,
                hidden_dims=self.hparams.hidden_dims,
                latent_dim=self.hparams.latent_dim,
            )
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
