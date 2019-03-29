"""Loss functions for fnet models."""


import torch


class HeteroscedasticLoss(torch.nn.Module):
    """Loss function to capture heteroscedastic aleatoric uncertainty."""

    def forward(
            self,
            y_hat_batch: torch.Tensor,
            y_batch: torch.Tensor,
    ):
        """Calculates loss.

        Parameters
        ----------
        y_hat_batch
           Batched, 2-channel model output.
        y_batch
           Batched, 1-channel target output.

        """
        mean_batch = y_hat_batch[:, 0:1, :, :, :]
        log_var_batch = y_hat_batch[:, 1:2, :, :, :]
        loss_batch = (
            0.5*torch.exp(-log_var_batch)*(mean_batch - y_batch).pow(2)
            + 0.5*log_var_batch
        )
        return loss_batch.mean()
