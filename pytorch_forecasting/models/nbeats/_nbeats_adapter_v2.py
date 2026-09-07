"""Shared N-Beats adapter for pytorch-forecasting v2."""

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._nbeats._blocks import (
    NBEATSSeasonalBlock,
    NBEATSTrendBlock,
)
from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class NBeatsAdapterV2(TslibBaseModel):
    """Shared forward and training logic for NBeats and NBeatsKAN (v2).

    This adapter bridges the v2 tslib batch format to the N-BEATS block
    computations originally implemented in the v1 ``NBeatsAdapter``.
    Subclasses are expected to build ``self.net_blocks`` (an
    ``nn.ModuleList``) during ``__init__``.

    Parameters
    ----------
    loss : Metric
        Loss function used for training.
    logging_metrics : list of nn.Module, optional
        Additional metrics logged during training/validation/test.
    optimizer : Optimizer or str, default="adam"
        Optimizer or name of a registered optimizer.
    optimizer_params : dict, optional
        Keyword arguments forwarded to the optimizer constructor.
    lr_scheduler : str, optional
        Name of a registered learning-rate scheduler.
    lr_scheduler_params : dict, optional
        Keyword arguments forwarded to the scheduler constructor.
    metadata : dict, optional
        DataModule metadata dict; used to extract ``context_length`` and
        ``prediction_length``.
    backcast_loss_ratio : float, default=0.0
        Weight of the backcast reconstruction loss relative to the
        forecast loss.  Set to 0 to disable.
    **kwargs : Any
        Additional keyword arguments forwarded to ``TslibBaseModel``.
    """

    def __init__(
        self,
        loss: Metric,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        backcast_loss_ratio: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
        )
        self.backcast_loss_ratio = backcast_loss_ratio

    def _target_from_batch(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract univariate target history from a v2 tslib batch.

        Parameters
        ----------
        x : dict of str to torch.Tensor
            Batch dictionary produced by ``TslibDataModule``.

        Returns
        -------
        torch.Tensor
            Target history tensor of shape ``(batch, context_length)``.
        """
        target = x["history_target"]
        if target.ndim == 3:
            target = target[..., 0]
        return target

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the N-BEATS forward pass.

        Parameters
        ----------
        x : dict of str to torch.Tensor
            Batch dictionary produced by ``TslibDataModule``.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary with keys ``"prediction"``, ``"backcast"``,
            ``"trend"``, ``"seasonality"``, and ``"generic"``.
        """
        target = self._target_from_batch(x)

        timesteps = self.context_length + self.prediction_length
        generic_forecast = [
            torch.zeros(
                (target.size(0), timesteps), dtype=torch.float32, device=self.device
            )
        ]
        trend_forecast = [
            torch.zeros(
                (target.size(0), timesteps), dtype=torch.float32, device=self.device
            )
        ]
        seasonal_forecast = [
            torch.zeros(
                (target.size(0), timesteps), dtype=torch.float32, device=self.device
            )
        ]
        forecast = torch.zeros(
            (target.size(0), self.prediction_length),
            dtype=torch.float32,
            device=self.device,
        )

        backcast = target
        for block in self.net_blocks:
            backcast_block, forecast_block = block(backcast)

            full = torch.cat([backcast_block.detach(), forecast_block.detach()], dim=1)
            if isinstance(block, NBEATSTrendBlock):
                trend_forecast.append(full)
            elif isinstance(block, NBEATSSeasonalBlock):
                seasonal_forecast.append(full)
            else:
                generic_forecast.append(full)

            # Avoid in-place op so autograd graph is not corrupted.
            backcast = backcast - backcast_block
            forecast = forecast + forecast_block

        prediction = forecast.unsqueeze(-1)
        backcast_out = (target - backcast).unsqueeze(-1)
        trend = torch.stack(trend_forecast, dim=0).sum(0).unsqueeze(-1)
        seasonality = torch.stack(seasonal_forecast, dim=0).sum(0).unsqueeze(-1)
        generic = torch.stack(generic_forecast, dim=0).sum(0).unsqueeze(-1)

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])
            backcast_out = self.transform_output(backcast_out, x["target_scale"])
            trend = self.transform_output(trend, x["target_scale"])
            seasonality = self.transform_output(seasonality, x["target_scale"])
            generic = self.transform_output(generic, x["target_scale"])

        return {
            "prediction": prediction,
            "backcast": backcast_out,
            "trend": trend,
            "seasonality": seasonality,
            "generic": generic,
        }

    def _compute_loss(
        self,
        x: dict[str, torch.Tensor],
        y: torch.Tensor,
        out: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute forecast loss with optional backcast regularisation.

        Parameters
        ----------
        x : dict of str to torch.Tensor
            Input batch dictionary.
        y : torch.Tensor
            Ground-truth forecast target.
        out : dict of str to torch.Tensor
            Output of :meth:`forward`.

        Returns
        -------
        tuple of torch.Tensor
            ``(loss, y_hat)`` where ``loss`` is the scalar training loss and
            ``y_hat`` is the raw prediction tensor.
        """
        y_hat = out["prediction"]
        loss = self.loss(y_hat, y)

        if self.backcast_loss_ratio > 0:
            backcast = out["backcast"].squeeze(-1)
            encoder_target = self._target_from_batch(x)

            backcast_weight = (
                self.backcast_loss_ratio
                * self.prediction_length
                / max(self.context_length, 1)
            )
            backcast_weight = backcast_weight / (backcast_weight + 1)
            forecast_weight = 1 - backcast_weight

            backcast_loss = (backcast - encoder_target).abs().mean() * backcast_weight
            loss = loss * forecast_weight + backcast_loss

        return loss, y_hat

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Run one training step.

        Parameters
        ----------
        batch : tuple of dict of str to torch.Tensor
            Batch produced by the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary with key ``"loss"``.
        """
        x, y = batch
        out = self(x)
        loss, y_hat = self._compute_loss(x, y, out)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="train")
        return {"loss": loss}

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Run one validation step.

        Parameters
        ----------
        batch : tuple of dict of str to torch.Tensor
            Batch produced by the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary with key ``"val_loss"``.
        """
        x, y = batch
        out = self(x)
        loss, y_hat = self._compute_loss(x, y, out)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="val")
        return {"val_loss": loss}

    def test_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Run one test step.

        Parameters
        ----------
        batch : tuple of dict of str to torch.Tensor
            Batch produced by the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary with key ``"test_loss"``.
        """
        x, y = batch
        out = self(x)
        loss, y_hat = self._compute_loss(x, y, out)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="test")
        return {"test_loss": loss}
