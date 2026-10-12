"""Export a trained RTEIR checkpoint to ONNX for TensorRT deployment.

This script loads the real checkpoint referenced by an options yml, so the
exported graph is the one used for the PSNR validation.

Layout:
    <out-dir>/<base>.onnx        fp32 weights, float32 I/O, static shape, opset 17
    <out-dir>/<base>_fp16.onnx   fp16 weights, float32 I/O

The RTEIR graph pads internally to a multiple of 8 (vit_window_size=8), so any
input size that is a multiple of 8 is supported directly.

NOTE: the *_fp16.onnx file produced here (onnxconverter_common) must NOT be
used to build TensorRT engines -- it yields NaN engines. Build the engine from
the fp32 ONNX with the fp16 flag (see scripts/trt_build_mixed_fp16.py or
scripts/trt_infer.py --fp16).

Usage:
    python scripts/export_onnx.py \
        --yml options/test/gopro.yml \
        --height 720 --width 1280 \
        --out-dir deploy_onnx --base rteir_gopro-720x1280
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from basicsr.models.archs import define_network  # noqa: E402


def load_net_from_yml(yml_path):
    with open(yml_path) as f:
        opt = yaml.safe_load(f)
    net_opt = copy.deepcopy(opt['network_g'])
    ckpt = opt['path']['pretrain_network_g']
    if ckpt is None:
        raise ValueError(f'{yml_path} has no pretrain_network_g')
    ckpt = (ROOT / ckpt[2:]) if ckpt.startswith('./') else Path(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    arch_type = net_opt.get('type')
    net = define_network(net_opt)
    state = torch.load(str(ckpt), map_location='cpu')
    if isinstance(state, dict) and 'params' in state:
        state = state['params']
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[len('module.'):]: v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f'  yml    : {yml_path}')
    print(f'  ckpt   : {ckpt}')
    print(f'  arch   : {arch_type}  params={n_params / 1e6:.3f}M  '
          f'event_ch={net_opt.get("inChannels_event", 6)}')
    return net, net_opt.get('inChannels_event', 6)


def export_onnx(net, h, w, event_channels, out_path):
    x = torch.randn(1, 3, h, w)
    event = torch.randn(1, event_channels, h, w)
    with torch.no_grad():
        out = net(x, event)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        assert tuple(out.shape[-2:]) == (h, w), out.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            net, (x, event), str(out_path),
            input_names=['x', 'event'], output_names=['output'],
            opset_version=17, do_constant_folding=True,
        )
    print(f'  ONNX   : {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)')


def simplify_onnx(path):
    try:
        import onnx
        from onnxsim import simplify
        m = onnx.load(str(path))
        m_sim, check = simplify(m)
        if not check:
            raise RuntimeError('onnxsim verification failed')
        onnx.save(m_sim, str(path))
        print(f'  simpl. : {path.name} -> {path.stat().st_size / 1e6:.1f} MB')
    except Exception as e:
        print(f'  simplify skipped: {e}')


def to_fp16(path):
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(str(path))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    p16 = path.with_name(path.stem + '_fp16.onnx')
    onnx.save(m16, str(p16))
    print(f'  FP16   : {p16.name} ({p16.stat().st_size / 1e6:.1f} MB)')
    return p16


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--yml', required=True)
    ap.add_argument('--height', type=int, required=True)
    ap.add_argument('--width', type=int, required=True)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--base', type=str, default=None,
                    help='file stem; default: <yml name>-<H>x<W>')
    ap.add_argument('--no-simplify', action='store_true')
    args = ap.parse_args()

    base = args.base or f'{Path(args.yml).stem}-{args.height}x{args.width}'
    out_dir = Path(args.out_dir).expanduser()
    fp32_path = out_dir / f'{base}.onnx'

    net, ev_ch = load_net_from_yml(args.yml)
    export_onnx(net, args.height, args.width, ev_ch, fp32_path)
    if not args.no_simplify:
        simplify_onnx(fp32_path)
    to_fp16(fp32_path)
    print('done.')


if __name__ == '__main__':
    main()
