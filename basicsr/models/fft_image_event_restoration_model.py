"""ImageEventRestorationModel with FFT loss support.

Identical to ImageEventRestorationModel but also reads ``fft_loss_opt``
from the training options and adds the frequency-domain loss during
optimize_parameters.
"""

import importlib
from collections import OrderedDict
from copy import deepcopy

from basicsr.models.image_event_restoration_model import ImageEventRestorationModel
from basicsr.utils import get_root_logger

loss_module = importlib.import_module("basicsr.models.losses")


class ImageEventRestorationModelFFT(ImageEventRestorationModel):
    """Extension that adds FFT (frequency domain) loss on top of pixel loss."""

    def init_training_settings(self):
        # Let the base class set up pixel / perceptual losses first.
        super().init_training_settings()

        train_opt = self.opt["train"]

        # ---- FFT loss ----
        if train_opt.get("fft_loss_opt"):
            fft_opt = deepcopy(train_opt["fft_loss_opt"])
            fft_type = fft_opt.pop("type")
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**fft_opt).to(self.device)
            logger = get_root_logger()
            logger.info(f"FFT loss enabled: {fft_type}")
        else:
            self.cri_fft = None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        if self.opt["datasets"]["train"].get("use_mask"):
            preds = self.net_g(x=self.lq, event=self.voxel, mask=self.mask)
        else:
            preds = self.net_g(x=self.lq, event=self.voxel)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()

        # ---- pixel loss (Charbonnier etc.) ----
        if self.cri_pix:
            l_pix = 0.0
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix

        # ---- FFT loss ----
        if self.cri_fft:
            l_fft = 0.0
            for pred in preds:
                l_fft += self.cri_fft(pred, self.gt)
            l_total += l_fft
            loss_dict["l_fft"] = l_fft

        # dummy term to keep DDP happy (unused params get gradient)
        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt["train"].get("use_grad_clip", True)
        if use_grad_clip:
            grad_clip_val = self.opt["train"].get("grad_clip_val", 0.01)
            from torch.nn.utils import clip_grad_norm_
            clip_grad_norm_(self.net_g.parameters(), grad_clip_val)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
