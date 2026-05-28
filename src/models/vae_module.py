from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAUROC

from .components.vae import VariationalAutoEncoder


def _elbo(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, recon_loss, kl_loss) — all scalar means over the batch."""
    recon = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)          # (B,)
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1)    # (B,)
    return (recon + beta * kl).mean(), recon.mean(), kl.mean()


class VAELitModule(LightningModule):
    """LightningModule for unsupervised anomaly detection with a Variational Autoencoder.

    Trains on background-only events (ELBO loss = reconstruction + β·KL).
    Anomaly score options:
      - "reconstruction": per-event MSE (default, robust for AD)
      - "elbo": per-event reconstruction + β·KL (captures latent-space surprise too)
    AUROC is computed on val/test sets containing background + signal.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        beta: float,
        anomaly_score_type: str,
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
        self.train_recon = MeanMetric()
        self.train_kl = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_recon = MeanMetric()
        self.val_kl = MeanMetric()

        self.test_loss = MeanMetric()

        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        self.val_auroc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.net(x)

    def _anomaly_score(self, x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Per-event anomaly score (higher = more anomalous)."""
        recon = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)
        if self.hparams.anomaly_score_type == "elbo":
            kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1)
            return recon + self.hparams.beta * kl
        return recon  # "reconstruction" (default)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_auroc.reset()
        self.val_auroc_best.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x_hat, mu, log_var = self.forward(x)
        loss, recon, kl = _elbo(x, x_hat, mu, log_var, self.hparams.beta)

        self.train_loss(loss)
        self.train_recon(recon)
        self.train_kl(kl)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recon", self.train_recon, on_step=False, on_epoch=True)
        self.log("train/kl", self.train_kl, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        x_hat, mu, log_var = self.forward(x)
        loss, recon, kl = _elbo(x, x_hat, mu, log_var, self.hparams.beta)

        self.val_loss(loss)
        self.val_recon(recon)
        self.val_kl(kl)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon", self.val_recon, on_step=False, on_epoch=True)
        self.log("val/kl", self.val_kl, on_step=False, on_epoch=True)

        scores = self._anomaly_score(x, x_hat, mu, log_var)
        self.val_auroc.update(scores, y)

    def on_validation_epoch_end(self) -> None:
        try:
            auroc = self.val_auroc.compute()
        except Exception:
            auroc = torch.tensor(0.0, device=self.device)
        self.val_auroc.reset()

        self.val_auroc_best(auroc)
        self.log("val/auroc", auroc, prog_bar=True)
        self.log("val/auroc_best", self.val_auroc_best.compute(), prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        x_hat, mu, log_var = self.forward(x)
        loss, _, _ = _elbo(x, x_hat, mu, log_var, self.hparams.beta)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        scores = self._anomaly_score(x, x_hat, mu, log_var)
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
            self.net = VariationalAutoEncoder(
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
