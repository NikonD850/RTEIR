"""ImageEventRestorationModel with FFT loss + EMA weights.

Extends ImageEventRestorationModelFFT:
- maintains an exponential moving average of net_g weights
- validation runs with EMA weights (swapped in/out around test())
- EMA weights are saved as net_g_ema_<iter>.pth alongside net_g

Single-stage training only: EMA is updated online during training.
"""

from copy import deepcopy

import torch

from basicsr.models.fft_image_event_restoration_model import (
    ImageEventRestorationModelFFT,
)
from basicsr.utils import get_root_logger


class ImageEventRestorationModelFFTEMA(ImageEventRestorationModelFFT):
    """FFT-loss model with online EMA weight averaging."""

    def init_training_settings(self):
        super().init_training_settings()

        self.ema_decay = self.opt["train"].get("ema_decay", 0.999)
        self.ema_warmup_iter = self.opt["train"].get("ema_warmup_iter", 1000)
        # deepcopy the raw module, NOT the DDP wrapper: a deepcopied DDP object
        # produces state_dict keys with a "module." prefix, which breaks the
        # key alignment in _update_ema / test below.
        net_raw = self._unwrap_ddp(self.net_g)
        self.net_g_ema = deepcopy(net_raw).eval().requires_grad_(False)

        logger = get_root_logger()
        logger.info(
            f"EMA enabled: decay={self.ema_decay}, warmup={self.ema_warmup_iter}"
        )

    @staticmethod
    def _unwrap_ddp(net):
        return net.module if hasattr(net, "module") else net

    @torch.no_grad()
    def _update_ema(self, current_iter):
        decay = (
            self.ema_decay
            if current_iter > self.ema_warmup_iter
            else min(1.0, 0.5 * (1 + current_iter / max(1, self.ema_warmup_iter)))
        )
        msd = self._unwrap_ddp(self.net_g).state_dict()
        for k, v in self.net_g_ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(msd[k].detach(), alpha=1 - decay)
            else:
                v.copy_(msd[k])

    def optimize_parameters(self, current_iter):
        super().optimize_parameters(current_iter)
        if hasattr(self, "net_g_ema"):
            self._update_ema(current_iter)

    def test(self):
        if not (self.is_train and hasattr(self, "net_g_ema")):
            return super().test()

        # swap in EMA weights on the raw module (DDP forward shares these
        # tensors), run base test (padding logic), swap back
        net_raw = self._unwrap_ddp(self.net_g)
        backup = {k: v.detach().clone() for k, v in net_raw.state_dict().items()}
        net_raw.load_state_dict(self.net_g_ema.state_dict())
        try:
            super().test()
        finally:
            net_raw.load_state_dict(backup)

    def save(self, epoch, current_iter):
        super().save(epoch, current_iter)
        if hasattr(self, "net_g_ema"):
            self.save_network(self.net_g_ema, "net_g_ema", current_iter)
