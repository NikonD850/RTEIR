import numpy as np
import torch


def events_to_voxel_grid(events, num_bins, width, height, return_format='CHW'):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param return_format: 'CHW' or 'HWC'
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    # print('DEBUG: voxel.shape:{}'.format(voxel_grid.shape))

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    # print('last stamp:{}'.format(last_stamp))
    # print('max stamp:{}'.format(events[:, 0].max()))
    # print('timestamp:{}'.format(events[:, 0]))
    # print('polarity:{}'.format(events[:, -1]))

    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT # 
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins # [True True ... True]
    # print('x max:{}'.format(xs[valid_indices].max()))
    # print('y max:{}'.format(ys[valid_indices].max()))
    # print('tix max:{}'.format(tis[valid_indices].max()))

    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width  ## ! ! !
            + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    if return_format == 'CHW':
        return voxel_grid
    elif return_format == 'HWC':
        return voxel_grid.transpose(1,2,0)

def voxel_norm(voxel):
    """
    Norm the voxel

    :param voxel: The unnormed voxel grid
    :return voxel: The normed voxel grid
    """
    nonzero_ev = (voxel != 0)
    num_nonzeros = nonzero_ev.sum()
    # print('DEBUG: num_nonzeros:{}'.format(num_nonzeros))
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = voxel.sum() / num_nonzeros
        stddev = torch.sqrt((voxel ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        voxel = mask * (voxel - mean) / stddev

    return voxel



def filter_event(x,y,p,t, s_e_index=[0,6]):
    '''
    s_e_index: include both left and right index
    '''
    t_1=t.squeeze(1)
    uniqw, inverse = np.unique(t_1, return_inverse=True)
    discretized_ts = np.bincount(inverse)
    index_exposure_start = np.sum(discretized_ts[0:s_e_index[0]])
    index_exposure_end = np.sum(discretized_ts[0:s_e_index[1]+1])
    x_1 = x[index_exposure_start:index_exposure_end]
    y_1 = y[index_exposure_start:index_exposure_end]
    p_1 = p[index_exposure_start:index_exposure_end]
    t_1 = t[index_exposure_start:index_exposure_end]
    
    return x_1, y_1, p_1, t_1



def voxel_norm_mass(voxel, eps=0.5):
    """Per-pixel mass normalization that preserves sign cancellation.

    Divides the signed voxel by the per-pixel total |event| mass, turning it
    into a unit-mass signed temporal distribution. Unlike the nonzero z-score
    in `voxel_norm` (which removes a per-sample scale), every pixel keeps its
    own relative magnitudes and the cancellative sign structure survives.
    """
    mass = voxel.abs().sum(dim=0, keepdim=True)
    return voxel / (mass + eps)


def voxel_norm_perbin(voxel, eps=1e-3):
    """Z-score every temporal bin independently instead of pooling all bins.

    `voxel_norm` derives one (mu, sigma) from all non-zero entries of the whole
    tensor, so the 12 bins -- whose event densities differ a lot, motion being
    concentrated in a few of them -- share a single scale and the busiest bin
    effectively sets it for all of them. This variant standardizes each channel
    (== each temporal bin) on its own, which keeps the "normalized" semantics
    while removing the cross-bin scale imbalance. The variance is clamped at 0
    because a per-bin count is small enough for `sum(x^2)/n - mean^2` to go
    slightly negative in float32.
    """
    out = voxel.clone()
    for b in range(voxel.shape[0]):
        x = voxel[b]
        nonzero_ev = (x != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = x.sum() / num_nonzeros
            var = ((x ** 2).sum() / num_nonzeros - mean ** 2).clamp_min(0.0)
            out[b] = nonzero_ev.float() * (x - mean) / torch.sqrt(var).clamp_min(eps)
    return out


# ---------------------------------------------------------------------------
# Global-constant-sigma normalization (review md §15.4, P3).
#
# Pooled non-zero std of the GoPro 12-bin signed voxel, measured with
# scripts/estimate_global_voxel_sigma.py over 240 train / 240 test frames:
#     train 0.9057   test 0.9028   (absmax ~6.8)
# The two splits agree to 0.3%, so this is a property of the dataset, not of
# the sampling -- exactly what a "constant" needs to be.
GO_PRO_12BIN_GLOBAL_SIGMA = 0.9057


def voxel_norm_gstd(voxel, sigma=GO_PRO_12BIN_GLOBAL_SIGMA):
    """Divide the signed voxel by a dataset-level constant sigma.

    Unlike `voxel_norm` (per-sample non-zero z-score) this uses no per-sample
    statistic at all: the same constant is applied to every crop and to the full
    frames used at validation, so there is no train/val statistic mismatch and
    no sample-dependent rescaling. Sign, zero point and all pixel-to-pixel
    relative magnitudes survive; only the absolute unit changes.
    """
    return voxel / float(sigma)


def voxel_norm_log(voxel, c=1.0):
    """Sign-preserving log compression -- the middle ground between no
    normalization at all and a full z-score.

    Keeps the sign (polarity), the ordering and the fact that magnitude is
    meaningful, while taming the heavy tail of raw event counts (measured
    absmax 5-15). No per-sample statistic is involved, so it is also immune to
    the train-crop / val-full-image statistic mismatch.
    """
    return torch.sign(voxel) * c * torch.log1p(voxel.abs() / c)
