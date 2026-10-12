"""Evaluate engine accuracy.

Usage:
    python scripts/trt_psnr_eval.py --dataset gopro \
        --engine deploy_onnx/rteir_gopro-720x1280_fp16.engine --ssim
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import tensorrt as trt  # noqa: E402
from basicsr.data.event_util import voxel_norm  # noqa: E402
from basicsr.metrics import calculate_psnr, calculate_ssim  # noqa: E402

DATASETS = {
    'gopro': {
        'blur': './datasets/GoPro/test/blur',
        'gt': './datasets/GoPro/test/gt',
        'voxel': './datasets/GoPro/test/voxel',
        'size': (720, 1280),
    },
}


# --------------------------------------------------------------------------- #
# Sample enumeration
# --------------------------------------------------------------------------- #

def list_folder_samples(ds):
    blur_root, gt_root, vox_root = Path(ds['blur']), Path(ds['gt']), Path(ds['voxel'])
    out = []
    for blur_path in sorted(blur_root.rglob('*.png')):
        rel = blur_path.relative_to(blur_root)
        gt_path = gt_root / rel
        vox_path = vox_root / rel.with_suffix('.npz')
        if not gt_path.is_file() or not vox_path.is_file():
            raise FileNotFoundError(f'pair missing for {rel}: gt={gt_path.is_file()} vox={vox_path.is_file()}')
        out.append({'blur': str(blur_path), 'gt': str(gt_path), 'voxel': str(vox_path)})
    return out


# --------------------------------------------------------------------------- #
# Data loading (basicsr protocol)
# --------------------------------------------------------------------------- #

def load_x(path):
    """BGR png -> RGB float32 CHW (img2tensor default bgr2rgb=True)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f'cannot read {path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def load_voxel(path):
    vox = np.load(str(path))['voxel'].astype(np.float32)  # HWC
    if vox.shape[0] in (6, 10, 12) and vox.shape[-1] not in (6, 10, 12):
        vox = vox.transpose(1, 2, 0)
    chw = torch.from_numpy(np.ascontiguousarray(vox.transpose(2, 0, 1)))
    return voxel_norm(chw).numpy()


# --------------------------------------------------------------------------- #
# TensorRT runner
# --------------------------------------------------------------------------- #

class TRTRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('failed to deserialize engine (version mismatch?)')
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            buf = torch.empty(shape, dtype=torch.float32, device='cuda')
            self.context.set_tensor_address(name, buf.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = buf
            else:
                self.outputs[name] = buf
        self.stream = torch.cuda.Stream()
        print(f'engine I/O: in={ {k: tuple(v.shape) for k, v in self.inputs.items()} } '
              f'out={ {k: tuple(v.shape) for k, v in self.outputs.items()} }')

    def infer(self, x_chw, ev_chw):
        self.inputs['x'].copy_(torch.from_numpy(x_chw).unsqueeze(0))
        self.inputs['event'].copy_(torch.from_numpy(ev_chw).unsqueeze(0))
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        name = next(iter(self.outputs))
        return self.outputs[name][0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--engine', required=True)
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--ssim', action='store_true')
    ap.add_argument('--out-json', type=str, default=None)
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    samples = list_folder_samples(ds)
    if args.max_samples:
        samples = samples[:args.max_samples]

    h, w = ds['size']
    print(f'dataset  : {args.dataset} ({h}x{w})')
    print(f'engine   : {args.engine}')
    print(f'samples  : {len(samples)}')

    runner = TRTRunner(args.engine)

    psnrs, ssims = [], []
    t0 = time.time()
    for i, sample in enumerate(samples):
        x_chw = load_x(sample['blur'])
        ev_chw = load_voxel(sample['voxel'])
        gt_u8 = cv2.imread(sample['gt'], cv2.IMREAD_COLOR)
        out = runner.infer(x_chw, ev_chw)
        if not np.isfinite(out).all():
            raise RuntimeError(f'non-finite network output at sample {i} ({sample})')
        # network output is RGB; tensor2img(rgb2bgr=True) flips it back, so flip
        # here as well to compare against the raw (cv2/BGR-ordered) GT image
        sr_u8 = np.clip(out, 0, 1).transpose(1, 2, 0)[..., ::-1] * 255.0
        sr_u8 = np.ascontiguousarray(sr_u8.round().astype(np.uint8))
        psnrs.append(calculate_psnr(sr_u8, gt_u8, crop_border=0, test_y_channel=False))
        if args.ssim:
            ssims.append(calculate_ssim(sr_u8, gt_u8, crop_border=0, test_y_channel=False))
        if (i + 1) % 50 == 0 or i == len(samples) - 1:
            msg = f'  [{i + 1}/{len(samples)}] avg PSNR: {np.mean(psnrs):.4f}'
            if args.ssim:
                msg += f'  avg SSIM: {np.mean(ssims):.4f}'
            print(msg, flush=True)

    dt = time.time() - t0
    res = {'dataset': args.dataset, 'engine': str(args.engine), 'samples': len(psnrs),
           'psnr': float(np.mean(psnrs))}
    if args.ssim:
        res['ssim'] = float(np.mean(ssims))
    print('=' * 60)
    print(f'TensorRT FP16 ({args.dataset}): PSNR={res["psnr"]:.4f}', end='')
    if args.ssim:
        print(f'  SSIM={res["ssim"]:.4f}', end='')
    print(f'  N={res["samples"]}  ({dt:.1f}s, {1000 * dt / max(1, len(psnrs)):.1f} ms/img)')

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'saved: {args.out_json}')


if __name__ == '__main__':
    main()
