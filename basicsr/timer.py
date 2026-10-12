import argparse
import copy
import csv
import inspect
import json
import math
import os
import re
import statistics
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

# Avoid CUDA OOM caused by memory fragmentation on large models
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn.functional as F
import yaml

warnings.filterwarnings('ignore', category=FutureWarning)

DEFAULT_MODELS = [
    'RTEIR',
]

IMAGE_NAMES = {
    'x', 'img', 'image', 'rgb', 'lq', 'frame', 'input', 'inp', 'x_img',
    'input_img', 'image_in'
}
EVENT_NAMES = {
    'event', 'events', 'voxel', 'voxels', 'event_voxel', 'x_event',
    'input_event', 'event_in'
}
MASK_NAMES = {'mask', 'Mask'}
STATE_NAMES = {'state', 'prev_state', 'hidden_state'}
SIZE_PRESETS = {
    '480p': (480, 854),
    '720p': (720, 1280),
    '1080p': (1080, 1920),
    '2k': (1440, 2560),
    '4k': (2160, 3840),
}


def parse_size(size_text):
    size_text = str(size_text).lower().strip()
    if size_text in SIZE_PRESETS:
        return SIZE_PRESETS[size_text]

    match = re.match(r'^(\d+)\s*[x,]\s*(\d+)$', size_text)
    if match is None:
        raise argparse.ArgumentTypeError(
            'size must be one of 720p/1080p or HxW, e.g. 720x1280')
    return int(match.group(1)), int(match.group(2))


def parse_key_values(items):
    parsed = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f'Expected key=value, but got {item!r}.')
        key, value = item.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f'Empty key in {item!r}.')
        parsed[key] = yaml.safe_load(value)
    return parsed


def parse_channels(items):
    channels = parse_key_values(items)
    return {str(key): int(value) for key, value in channels.items()}


def normalize_model_names(model_args):
    names = []
    for item in model_args or []:
        names.extend(part.strip() for part in item.split(',') if part.strip())
    return names


def import_define_network():
    from basicsr.models.archs import define_network
    return define_network


def load_yaml_options(opt_path):
    from basicsr.utils.options import ordered_yaml

    with open(opt_path, 'r') as file:
        Loader, _ = ordered_yaml()
        return yaml.load(file, Loader=Loader)


def discover_model_names():
    arch_dir = Path(__file__).resolve().parent / 'models' / 'archs'
    alias_pattern = re.compile(
        r'globals\(\)\[[\"\'](.+?)[\"\']\]\s*=\s*([A-Za-z_][A-Za-z0-9_]*)')
    class_to_alias = {}
    for arch_file in sorted(arch_dir.glob('*_arch.py')):
        for line in arch_file.read_text().splitlines():
            match = alias_pattern.search(line)
            if match is None:
                continue
            alias, class_name = match.groups()
            old_alias = class_to_alias.get(class_name)
            prefer_new = old_alias is None or alias == class_name
            prefer_new = prefer_new or ('-' not in alias and '-' in old_alias)
            if prefer_new:
                class_to_alias[class_name] = alias
    return sorted(class_to_alias.values())


def build_network(model_name, base_network_opt, model_kwargs):
    define_network = import_define_network()
    network_opt = copy.deepcopy(base_network_opt) if base_network_opt else {}
    network_opt.update(model_kwargs)
    network_opt['type'] = model_name
    return define_network(copy.deepcopy(network_opt)), network_opt


def load_checkpoint(net, checkpoint_path, param_key='auto', strict=True):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if param_key != 'auto':
            state_dict = checkpoint[param_key]
        else:
            state_dict = None
            for key in ('params_ema', 'params', 'state_dict'):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError('Checkpoint does not contain a state dict.')

    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith('module.') else key
        clean_state_dict[clean_key] = value
    return net.load_state_dict(clean_state_dict, strict=strict)


def count_parameters(net):
    return sum(param.numel() for param in net.parameters())


def format_count(value):
    if value >= 1e9:
        return f'{value / 1e9:.3f}B'
    if value >= 1e6:
        return f'{value / 1e6:.3f}M'
    if value >= 1e3:
        return f'{value / 1e3:.3f}K'
    return str(value)


def effective_hw(height, width, pad_to):
    if pad_to <= 1:
        return height, width
    return math.ceil(height / pad_to) * pad_to, math.ceil(width / pad_to) * pad_to


def get_forward_parameters(net):
    signature = inspect.signature(net.forward)
    params = []
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        params.append(param)
    return params


def is_optional(param):
    return param.default is not inspect.Parameter.empty


def classify_input(name):
    lower_name = name.lower()
    if name in MASK_NAMES or lower_name in {item.lower() for item in MASK_NAMES}:
        return 'mask'
    if lower_name in EVENT_NAMES:
        return 'event'
    if lower_name in STATE_NAMES:
        return 'state'
    if lower_name in IMAGE_NAMES:
        return 'image'
    if 'event' in lower_name or 'voxel' in lower_name:
        return 'event'
    if 'mask' in lower_name:
        return 'mask'
    if 'state' in lower_name:
        return 'state'
    if 'img' in lower_name or 'image' in lower_name or lower_name in {'x', 'inp'}:
        return 'image'
    return 'extra'


def select_input_params(net, include_optional_mask=False):
    selected = []
    seen_kinds = set()
    for param in get_forward_parameters(net):
        kind = classify_input(param.name)
        if is_optional(param):
            # Include optional image/event params (e.g. BDHNet: x=None, event=None)
            # But skip aliases (e.g. x_blur when x already selected)
            if kind in ('image', 'event') and kind not in seen_kinds:
                selected.append((param.name, kind))
                seen_kinds.add(kind)
            elif kind == 'mask' and include_optional_mask:
                selected.append((param.name, kind))
            # Skip optional state/extra params
            continue
        if kind not in seen_kinds:
            selected.append((param.name, kind))
            seen_kinds.add(kind)
    return selected


def make_tensor(kind, channels, batch_size, height, width, dtype, pin_memory):
    if kind == 'mask':
        tensor = torch.ones(batch_size, channels, height, width, dtype=dtype)
    else:
        tensor = torch.randn(batch_size, channels, height, width, dtype=dtype)

    if pin_memory:
        try:
            tensor = tensor.pin_memory()
        except RuntimeError:
            pass
    return tensor


def make_cpu_inputs(input_params, args, height, width, dtype, pin_memory):
    inputs = []
    descriptions = []
    channel_overrides = args.input_channels
    for name, kind in input_params:
        if kind == 'state':
            inputs.append(None)
            descriptions.append(f'{name}=None')
            continue

        if name in channel_overrides:
            channels = channel_overrides[name]
        elif kind == 'image':
            channels = args.img_channels
        elif kind == 'event':
            channels = args.event_channels
        elif kind == 'mask':
            channels = args.mask_channels
        else:
            channels = args.extra_channels

        inputs.append(
            make_tensor(kind, channels, args.batch_size, height, width, dtype,
                        pin_memory))
        descriptions.append(f'{name}:{channels}x{height}x{width}')
    return inputs, descriptions


def move_inputs_to_device(inputs, device, non_blocking):
    moved = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device, non_blocking=non_blocking))
        else:
            moved.append(value)
    return moved


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def copy_all_tensors_to_cpu(value, non_blocking):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', non_blocking=non_blocking)
    if isinstance(value, dict):
        return {
            key: copy_all_tensors_to_cpu(item, non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(copy_all_tensors_to_cpu(item, non_blocking) for item in value)
    if isinstance(value, list):
        return [copy_all_tensors_to_cpu(item, non_blocking) for item in value]
    return value


def pad_tensor_to_hw(tensor, height, width):
    if not isinstance(tensor, torch.Tensor) or tensor.dim() < 4:
        return tensor
    pad_h = height - tensor.shape[-2]
    pad_w = width - tensor.shape[-1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f'Cannot pad tensor from {tuple(tensor.shape[-2:])} to '
            f'{height}x{width}.')
    if pad_h == 0 and pad_w == 0:
        return tensor
    pad_mode = 'reflect' if pad_h < tensor.shape[-2] and pad_w < tensor.shape[-1] else 'replicate'
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=pad_mode)


def pad_inputs_to_hw(inputs, height, width):
    return [pad_tensor_to_hw(value, height, width) for value in inputs]


def crop_spatial_tensors(value, height, width):
    if isinstance(value, torch.Tensor):
        if value.dim() >= 4 and value.shape[-2] >= height and value.shape[-1] >= width:
            return value[..., :height, :width]
        return value
    if isinstance(value, dict):
        return {
            key: crop_spatial_tensors(item, height, width)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(crop_spatial_tensors(item, height, width) for item in value)
    if isinstance(value, list):
        return [crop_spatial_tensors(item, height, width) for item in value]
    return value


def copy_output_to_cpu(output, mode, non_blocking, to_numpy):
    if mode == 'none':
        return None
    if mode == 'all':
        cpu_output = copy_all_tensors_to_cpu(output, non_blocking)
    else:
        output_tensor = first_tensor(output)
        if output_tensor is None:
            return None
        cpu_output = output_tensor.detach().to('cpu', non_blocking=non_blocking)

    if to_numpy and isinstance(cpu_output, torch.Tensor):
        return cpu_output.numpy()
    return cpu_output


def sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def cuda_memory_mb(device):
    if device.type != 'cuda':
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def reset_cuda_memory(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def autocast_context(args, device):
    if not args.amp or device.type != 'cuda':
        return nullcontext()
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float8': torch.float8_e4m3fn,
    }
    dtype = dtype_map.get(args.amp_dtype, torch.float16)
    return torch.autocast(device_type='cuda', dtype=dtype)


def time_loop(callback, iterations, warmup, device):
    """Time `callback` for `iterations` reps after `warmup` reps.

    Protocol aligned with event-deblur-baseline-speedtest/timer.py: on CUDA the
    per-iteration latency is measured with torch.cuda.Event(enable_timing=True)
    (record -> forward -> record -> synchronize), the same method every baseline
    repo uses. On CPU it falls back to perf_counter.
    """
    times = []
    use_events = device.type == 'cuda'
    for index in range(warmup + iterations):
        if use_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = callback(index)
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            sync_device(device)
            start_time = time.perf_counter()
            result = callback(index)
            sync_device(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if index >= warmup:
            times.append(elapsed_ms)
        del result
    return times


def summarize_times(times, batch_size, trim_head=0, trim_tail=0):
    # Baseline protocol: discard the first `trim_head` and last `trim_tail`
    # timed runs, then take stats over the kept runs (default 40 of 50).
    end = len(times) - trim_tail if trim_tail > 0 else len(times)
    kept = times[trim_head:end]
    if not kept:
        kept = times
    mean_ms = statistics.mean(kept)
    median_ms = statistics.median(kept)
    # np.std (population, ddof=0) is what the baseline timers report.
    std_ms = statistics.pstdev(kept) if len(kept) > 1 else 0.0
    fps = batch_size * 1000.0 / mean_ms
    return {
        'mean_ms': mean_ms,
        'median_ms': median_ms,
        'std_ms': std_ms,
        'min_ms': min(kept),
        'max_ms': max(kept),
        'fps': fps,
        'kept': len(kept),
        'total': len(times),
    }


def maybe_compile(net, enabled):
    if not enabled:
        return net
    if not hasattr(torch, 'compile'):
        raise RuntimeError('torch.compile is not available in this PyTorch.')
    return torch.compile(net)


def benchmark_model(model_name, base_network_opt, checkpoint_path, args, device,
                    input_dtype):
    net, network_opt = build_network(model_name, base_network_opt,
                                     args.model_kwargs)
    load_info = None
    if checkpoint_path:
        incompatible = load_checkpoint(net, checkpoint_path, args.param_key,
                                       args.strict_load)
        load_info = {
            'missing_keys': list(getattr(incompatible, 'missing_keys', [])),
            'unexpected_keys': list(getattr(incompatible, 'unexpected_keys', [])),
        }

    input_params = select_input_params(net, args.include_mask)
    if not input_params:
        raise RuntimeError('Could not infer any forward inputs.')

    param_count = count_parameters(net)

    if args.dtype != 'float32' and not args.amp:
        net = net.to(dtype=input_dtype)
    net = maybe_compile(net, args.compile)
    net = net.to(device).eval()

    height, width = effective_hw(args.height, args.width, args.pad_to)
    pin_memory = args.pin_memory and device.type == 'cuda'
    cpu_inputs, input_desc = make_cpu_inputs(input_params, args, args.height,
                                             args.width, input_dtype,
                                             pin_memory)

    reset_cuda_memory(device)

    with torch.inference_mode():

        def run_e2e(_):
            device_inputs = move_inputs_to_device(cpu_inputs, device,
                                                  args.non_blocking)
            device_inputs = pad_inputs_to_hw(device_inputs, height, width)
            with autocast_context(args, device):
                output = net(*device_inputs)
            output = crop_spatial_tensors(output, args.height, args.width)
            return copy_output_to_cpu(output, args.copy_output,
                                      args.non_blocking, args.to_numpy)

        e2e_times = time_loop(run_e2e, args.frames, args.warmup, device)
        peak_memory = cuda_memory_mb(device)

        device_inputs = move_inputs_to_device(cpu_inputs, device,
                                              args.non_blocking)
        device_inputs = pad_inputs_to_hw(device_inputs, height, width)

        def run_h2d(_):
            return move_inputs_to_device(cpu_inputs, device, args.non_blocking)

        def run_forward(_):
            with autocast_context(args, device):
                return net(*device_inputs)

        forward_output = run_forward(0)

        def run_d2h(_):
            output = crop_spatial_tensors(forward_output, args.height,
                                          args.width)
            return copy_output_to_cpu(output, args.copy_output,
                                      args.non_blocking, args.to_numpy)

        h2d_times = time_loop(run_h2d, args.frames, args.warmup, device)
        forward_times = time_loop(run_forward, args.frames, args.warmup, device)
        d2h_times = time_loop(run_d2h, args.frames, args.warmup, device)

    params = param_count
    result = {
        'model': model_name,
        'network_opt': network_opt,
        'params': params,
        'params_readable': format_count(params),
        'device': str(device),
        'batch_size': args.batch_size,
        'requested_size': [args.height, args.width],
        'effective_size': [height, width],
        'frames': args.frames,
        'warmup': args.warmup,
        'input_desc': input_desc,
        'copy_output': args.copy_output,
        'pin_memory': pin_memory,
        'non_blocking': args.non_blocking,
        'amp': args.amp,
        'dtype': args.dtype,
        'peak_memory_mb': peak_memory,
        'e2e': summarize_times(e2e_times, args.batch_size,
                               args.trim_head, args.trim_tail),
        'h2d': summarize_times(h2d_times, args.batch_size,
                               args.trim_head, args.trim_tail),
        'forward': summarize_times(forward_times, args.batch_size,
                                   args.trim_head, args.trim_tail),
        'd2h': summarize_times(d2h_times, args.batch_size,
                               args.trim_head, args.trim_tail),
        'checkpoint': checkpoint_path,
        'load_info': load_info,
    }
    return result


def print_result(result):
    print('\n' + '=' * 80)
    print(f"Model: {result['model']}")
    print(f"Params: {result['params_readable']} ({result['params']:,d})")
    print(
        f"Device: {result['device']} | batch={result['batch_size']} | "
        f"frames={result['frames']} | warmup={result['warmup']}")
    print(
        f"Size: requested={result['requested_size'][0]}x{result['requested_size'][1]} "
        f"effective={result['effective_size'][0]}x{result['effective_size'][1]}")
    print('Inputs: ' + ', '.join(result['input_desc']))
    print(
        f"Output CPU copy: {result['copy_output']} | "
        f"peak CUDA memory: {result['peak_memory_mb']:.1f} MB")
    print('-' * 80)
    print(
        f"End-to-end (CPU->GPU + pad + forward + crop + GPU->CPU): "
        f"{result['e2e']['mean_ms']:.3f} ms/frame, "
        f"{result['e2e']['fps']:.2f} FPS")
    print(
        f"Forward only (inputs already on device): "
        f"{result['forward']['mean_ms']:.3f} +/- {result['forward']['std_ms']:.3f} ms/frame, "
        f"{result['forward']['fps']:.2f} FPS "
        f"(kept {result['forward']['kept']}/{result['forward']['total']}, FPS=1000/mean)")
    print('-' * 80)
    for key, label in [('h2d', 'CPU->GPU'), ('forward', 'Forward'),
                       ('d2h', 'GPU->CPU')]:
        item = result[key]
        print(
            f"{label:>9}: mean={item['mean_ms']:.3f} ms, "
            f"median={item['median_ms']:.3f} ms, "
            f"std={item['std_ms']:.3f} ms, "
            f"min={item['min_ms']:.3f} ms, max={item['max_ms']:.3f} ms")

    load_info = result.get('load_info')
    if load_info:
        print(
            f"Checkpoint load: missing={len(load_info['missing_keys'])}, "
            f"unexpected={len(load_info['unexpected_keys'])}")


def make_arg_parser():
    parser = argparse.ArgumentParser(
        description='Benchmark BasicSR arch inference FPS with transfer timing.')
    parser.add_argument(
        '--model', '--models', nargs='*', default=None,
        help='Model type(s). Use comma/space separated names, or all. '
        'Default: RTEIR.')
    parser.add_argument(
        '--list-models', action='store_true', help='List discoverable arch aliases.')
    parser.add_argument(
        '--opt', default=None,
        help='Optional yaml config. network_g is used as model kwargs.')
    parser.add_argument(
        '--model-kwargs', nargs='*', default=[],
        help='Extra network kwargs as key=value, e.g. scf_alpha=0.5.')
    parser.add_argument(
        '--checkpoint', default=None,
        help='Optional checkpoint. If omitted with --opt, path.pretrain_network_g is used.')
    parser.add_argument(
        '--param-key', default='auto',
        help='Checkpoint key: auto, params_ema, params, state_dict, ...')
    parser.add_argument(
        '--strict-load', action=argparse.BooleanOptionalAction, default=True,
        help='Use strict checkpoint loading. Default: true.')
    parser.add_argument(
        '--size', default='720p', type=parse_size,
        help='Input size preset or HxW. Default: 720p = 720x1280.')
    parser.add_argument('--height', type=int, default=None, help='Override input height.')
    parser.add_argument('--width', type=int, default=None, help='Override input width.')
    parser.add_argument('--pad-to', type=int, default=64,
                        help='Pad input H/W up to this multiple. Default: 64.')
    parser.add_argument('--frames', type=int, default=50,
                        help='Measured frames (timed iters). Default: 50 (baseline protocol).')
    parser.add_argument('--warmup', type=int, default=20,
                        help='Warmup frames not counted. Default: 20.')
    parser.add_argument('--trim-head', type=int, default=5,
                        help='Discard this many first timed runs. Default: 5 (baseline).')
    parser.add_argument('--trim-tail', type=int, default=5,
                        help='Discard this many last timed runs. Default: 5 (baseline).')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size.')
    parser.add_argument('--img-channels', type=int, default=3,
                        help='Default image input channels.')
    parser.add_argument('--event-channels', type=int, default=6,
                        help='Default event/voxel input channels.')
    parser.add_argument('--mask-channels', type=int, default=1,
                        help='Default mask input channels.')
    parser.add_argument('--extra-channels', type=int, default=3,
                        help='Channels for unknown required tensor inputs.')
    parser.add_argument(
        '--input-channels', nargs='*', default=[],
        help='Override forward input channels by name, e.g. x=3 event=12.')
    parser.add_argument('--include-mask', action='store_true',
                        help='Pass optional mask if forward has mask=None.')
    parser.add_argument('--device', default=None,
                        help='cuda, cuda:0, or cpu. Default: cuda if available.')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16', 'float8'],
                        default='float32', help='Tensor/model dtype when AMP is off.')
    parser.add_argument('--amp', action='store_true',
                        help='Use CUDA autocast for forward.')
    parser.add_argument('--amp-dtype', choices=['float16', 'bfloat16', 'float8'],
                        default='float16', help='Autocast dtype.')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile before benchmarking.')
    parser.add_argument('--tf32', action=argparse.BooleanOptionalAction,
                        default=True, help='Allow TF32 on CUDA. Default: true.')
    parser.add_argument('--cudnn-benchmark', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Enable the cuDNN autotuner (benchmark mode). Default: true.')
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction,
                        default=True, help='Pin CPU input tensors on CUDA. Default: true.')
    parser.add_argument('--non-blocking', action=argparse.BooleanOptionalAction,
                        default=True, help='Use non_blocking tensor copies. Default: true.')
    parser.add_argument(
        '--copy-output', choices=['first', 'all', 'none'], default='first',
        help='Which output tensors to copy back to CPU. Default: first.')
    parser.add_argument('--to-numpy', action='store_true',
                        help='Convert copied tensor output to numpy view.')
    parser.add_argument('--skip-failed', action='store_true',
                        help='Continue benchmarking if a model fails.')
    parser.add_argument('--save-json', default=None,
                        help='Optional path to save benchmark results as JSON.')
    parser.add_argument(
        '--options-dir', default=None,
        help='Batch mode: directory of YAML configs. '
        'Each YAML must define network_g and optionally path/pretrain_network_g, '
        'height, width, event_channels. '
        'Default directory: options-timer.')
    parser.add_argument(
        '--output-dir', default='results-timer',
        help='Output directory for CSV results. Default: results-timer.')
    return parser


CSV_HEADER = [
    'experiment', 'model', 'params', 'params_readable',
    'device', 'batch_size', 'height', 'width',
    'effective_height', 'effective_width',
    'img_channels', 'event_channels',
    'frames', 'warmup', 'amp', 'dtype',
    'peak_memory_mb',
    'e2e_mean_ms', 'e2e_median_ms', 'e2e_std_ms',
    'e2e_min_ms', 'e2e_max_ms', 'e2e_fps',
    'forward_mean_ms', 'forward_median_ms', 'forward_std_ms',
    'forward_min_ms', 'forward_max_ms', 'forward_fps',
    'h2d_mean_ms', 'd2h_mean_ms',
    'checkpoint',
]


def result_to_csv_row(result):
    """Convert a benchmark result dict to a flat CSV row dict."""
    return {
        'experiment': result.get('experiment', ''),
        'model': result['model'],
        'params': result['params'],
        'params_readable': result['params_readable'],
        'device': result['device'],
        'batch_size': result['batch_size'],
        'height': result['requested_size'][0],
        'width': result['requested_size'][1],
        'effective_height': result['effective_size'][0],
        'effective_width': result['effective_size'][1],
        'img_channels': result.get('img_channels', ''),
        'event_channels': result.get('event_channels', ''),
        'frames': result['frames'],
        'warmup': result['warmup'],
        'amp': result['amp'],
        'dtype': result['dtype'],
        'peak_memory_mb': f"{result['peak_memory_mb']:.1f}",
        'e2e_mean_ms': f"{result['e2e']['mean_ms']:.3f}",
        'e2e_median_ms': f"{result['e2e']['median_ms']:.3f}",
        'e2e_std_ms': f"{result['e2e']['std_ms']:.3f}",
        'e2e_min_ms': f"{result['e2e']['min_ms']:.3f}",
        'e2e_max_ms': f"{result['e2e']['max_ms']:.3f}",
        'e2e_fps': f"{result['e2e']['fps']:.2f}",
        'forward_mean_ms': f"{result['forward']['mean_ms']:.3f}",
        'forward_median_ms': f"{result['forward']['median_ms']:.3f}",
        'forward_std_ms': f"{result['forward']['std_ms']:.3f}",
        'forward_min_ms': f"{result['forward']['min_ms']:.3f}",
        'forward_max_ms': f"{result['forward']['max_ms']:.3f}",
        'forward_fps': f"{result['forward']['fps']:.2f}",
        'h2d_mean_ms': f"{result['h2d']['mean_ms']:.3f}",
        'd2h_mean_ms': f"{result['d2h']['mean_ms']:.3f}",
        'checkpoint': result.get('checkpoint', ''),
    }


def csv_filename(base_name, args):
    """Generate CSV filename with dtype suffix if not float32."""
    suffix = ''
    if args.dtype != 'float32':
        suffix = f'_{args.dtype}'
    return f'{base_name}{suffix}.csv'


def save_csv(rows, output_path):
    """Save a list of row dicts to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f'\nSaved CSV: {output_path}')


def run_single_opt(opt_path, args):
    """Run benchmark for a single YAML config file.

    Returns (results_list, failures_list).
    """
    opt = load_yaml_options(opt_path)
    experiment_name = Path(opt_path).stem

    # Allow YAML to override height/width/event_channels
    yaml_height = opt.get('height')
    yaml_width = opt.get('width')
    yaml_event_channels = opt.get('event_channels')
    yaml_img_channels = opt.get('img_channels')
    yaml_frames = opt.get('frames')
    yaml_warmup = opt.get('warmup')
    yaml_batch_size = opt.get('batch_size')

    saved_height = args.height
    saved_width = args.width
    saved_event_channels = args.event_channels
    saved_img_channels = args.img_channels
    saved_frames = args.frames
    saved_warmup = args.warmup
    saved_batch_size = args.batch_size

    if yaml_height is not None:
        args.height = yaml_height
    if yaml_width is not None:
        args.width = yaml_width
    if yaml_event_channels is not None:
        args.event_channels = yaml_event_channels
    if yaml_img_channels is not None:
        args.img_channels = yaml_img_channels
    if yaml_frames is not None:
        args.frames = yaml_frames
    if yaml_warmup is not None:
        args.warmup = yaml_warmup
    if yaml_batch_size is not None:
        args.batch_size = yaml_batch_size

    base_network_opt = copy.deepcopy(opt.get('network_g', {}))
    checkpoint_path = opt.get('path', {}).get('pretrain_network_g')

    model_names = normalize_model_names(args.model)
    all_mode = model_names and 'all' in {
        name.lower()
        for name in model_names
    }
    if all_mode:
        model_names = discover_model_names()
        args.skip_failed = True
    elif not model_names:
        if base_network_opt.get('type'):
            model_names = [base_network_opt['type']]
        else:
            model_names = list(DEFAULT_MODELS)

    base_network_opt.pop('type', None)

    device = torch.device(args.device)
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float8': torch.float16,  # float8 uses fp16 inputs + autocast
    }
    input_dtype = dtype_map[args.dtype]
    # Auto-enable AMP for float8
    if args.dtype == 'float8' and not args.amp:
        args.amp = True
        args.amp_dtype = 'float8'

    results = []
    failures = []
    for model_name in model_names:
        try:
            result = benchmark_model(model_name, base_network_opt,
                                     checkpoint_path, args, device, input_dtype)
            result['experiment'] = experiment_name
            result['img_channels'] = args.img_channels
            result['event_channels'] = args.event_channels
            results.append(result)
            print_result(result)
        except Exception as error:
            failures.append({
                'experiment': experiment_name,
                'model': model_name,
                'error': repr(error),
            })
            if not args.skip_failed:
                raise
            print(f'\n[SKIP] {model_name}: {error!r}')
        finally:
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Restore CLI overrides
    args.height = saved_height
    args.width = saved_width
    args.event_channels = saved_event_channels
    args.img_channels = saved_img_channels
    args.frames = saved_frames
    args.warmup = saved_warmup
    args.batch_size = saved_batch_size

    return results, failures


def main():
    parser = make_arg_parser()
    args = parser.parse_args()

    if args.list_models:
        for name in discover_model_names():
            print(name)
        return

    args.model_kwargs = parse_key_values(args.model_kwargs)
    args.input_channels = parse_channels(args.input_channels)

    parsed_height, parsed_width = args.size
    height_override = args.height
    width_override = args.width
    args.height = height_override if height_override is not None else parsed_height
    args.width = width_override if width_override is not None else parsed_width

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    # ---- Batch mode: --options-dir ----
    if args.options_dir is not None:
        options_dir = Path(args.options_dir)
        if not options_dir.is_dir():
            raise FileNotFoundError(
                f'options-dir not found: {options_dir}')
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        yaml_files = sorted(
            list(options_dir.glob('*.yml')) +
            list(options_dir.glob('*.yaml')))
        if not yaml_files:
            print(f'No YAML files found in {options_dir}')
            return

        print(f'Found {len(yaml_files)} config(s) in {options_dir}')
        all_results = []
        all_failures = []
        for yaml_file in yaml_files:
            print(f'\n{"=" * 80}')
            print(f'Processing: {yaml_file.name}')
            print(f'{"=" * 80}')
            results, failures = run_single_opt(yaml_file, args)
            all_results.extend(results)
            all_failures.extend(failures)

            # Save per-experiment CSV
            if results:
                rows = [result_to_csv_row(r) for r in results]
                csv_path = output_dir / csv_filename(yaml_file.stem, args)
                save_csv(rows, csv_path)

        # Save combined JSON
        json_path = output_dir / 'summary.json'
        json_path.write_text(
            json.dumps(
                {
                    'results': all_results,
                    'failures': all_failures,
                },
                indent=2,
                ensure_ascii=False,
                default=str))
        print(f'\nSaved summary JSON: {json_path}')

        if all_failures:
            print('\nFailures:')
            for item in all_failures:
                print(
                    f"- [{item['experiment']}] {item['model']}: "
                    f"{item['error']}")
        return

    # ---- Single mode: --opt or --model ----
    opt = load_yaml_options(args.opt) if args.opt else None
    base_network_opt = {}
    checkpoint_path = args.checkpoint
    if opt is not None:
        base_network_opt = copy.deepcopy(opt.get('network_g', {}))
        if not checkpoint_path:
            checkpoint_path = opt.get('path', {}).get('pretrain_network_g')

    model_names = normalize_model_names(args.model)
    all_mode = 'all' in {name.lower() for name in model_names}
    if all_mode:
        model_names = discover_model_names()
        args.skip_failed = True
    elif not model_names:
        if base_network_opt.get('type'):
            model_names = [base_network_opt['type']]
        else:
            model_names = list(DEFAULT_MODELS)

    base_network_opt.pop('type', None)

    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float8': torch.float16,  # float8 uses fp16 inputs + autocast
    }
    input_dtype = dtype_map[args.dtype]
    # Auto-enable AMP for float8
    if args.dtype == 'float8' and not args.amp:
        args.amp = True
        args.amp_dtype = 'float8'

    results = []
    failures = []
    for model_name in model_names:
        try:
            result = benchmark_model(model_name, base_network_opt,
                                     checkpoint_path, args, device, input_dtype)
            result['img_channels'] = args.img_channels
            result['event_channels'] = args.event_channels
            results.append(result)
            print_result(result)
        except Exception as error:
            failures.append({'model': model_name, 'error': repr(error)})
            if not args.skip_failed:
                raise
            print(f'\n[SKIP] {model_name}: {error!r}')
        finally:
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if args.save_json:
        output = {'results': results, 'failures': failures}
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f'\nSaved JSON: {output_path}')

    # Auto-save CSV to output-dir in single mode too
    if results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [result_to_csv_row(r) for r in results]
        exp_name = Path(args.opt).stem if args.opt else results[0]['model']
        csv_path = output_dir / csv_filename(exp_name, args)
        save_csv(rows, csv_path)

    if failures:
        print('\nFailures:')
        for item in failures:
            print(f"- {item['model']}: {item['error']}")


def entry_point():
    main()


if __name__ == '__main__':
    main()
