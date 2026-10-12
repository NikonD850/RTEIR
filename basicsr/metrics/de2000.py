"""CIEDE2000 color difference metric (average per-pixel deltaE2000).

Inputs are uint8 BGR numpy arrays (H, W, 3) in [0, 255], i.e. the channel
order produced by ``tensor2img`` with the default ``rgb2bgr=True`` (the order
cv2 / ``imwrite`` expects). They are converted to RGB inside
``calculate_de2000`` so the metric no longer depends on ``val.rgb2bgr``.
"""

import numpy as np


def _srgb_to_lab(img):
    """Convert uint8 sRGB image to CIELAB (D65). Returns HxWx3 float64."""
    x = np.asarray(img, dtype=np.float64) / 255.0
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)

    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = x @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])

    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4.0 / 29.0)

    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    lab = np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)
    return lab


def _bgr2rgb(img):
    """Swap first/last channel for 3-channel images; pass others through."""
    if img.ndim == 3 and img.shape[-1] == 3:
        return img[..., ::-1]
    return img


def calculate_de2000(img, gt, **kwargs):
    """Average CIEDE2000 color difference between img and gt (uint8 BGR)."""
    lab1 = _srgb_to_lab(_bgr2rgb(np.asarray(img)))
    lab2 = _srgb_to_lab(_bgr2rgb(np.asarray(gt)))

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dHp = 2.0 * np.sqrt(np.maximum(C1p * C2p, 0.0)) * np.sin(np.radians(dhp) / 2.0)

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0

    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hbarp = np.where(hdiff <= 180.0, hsum / 2.0, np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0))

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hbarp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hbarp))
        + 0.32 * np.cos(np.radians(3.0 * hbarp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hbarp - 63.0))
    )

    dtheta = 30.0 * np.exp(-((hbarp - 275.0) / 25.0) ** 2)
    Rc = 2.0 * np.sqrt(Cbarp**7 / (Cbarp**7 + 25.0**7))
    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc

    Sl = 1.0 + 0.015 * (Lbarp - 50.0) ** 2 / np.sqrt(20.0 + (Lbarp - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T

    dE = np.sqrt(
        (dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh)
    )
    return float(np.mean(dE))
