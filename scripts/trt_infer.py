"""TensorRT inference script for RTEIR.

Usage:
    # Build engine from ONNX and run inference benchmark
    python scripts/trt_infer.py \
        --onnx deploy_onnx/rteir_gopro-720x1280.onnx \
        --height 720 --width 1280 \
        --fp16 \
        --frames 60 --warmup 20 \
        --output-engine deploy_onnx/rteir_gopro-720x1280_fp16.engine

    # Use pre-built engine
    python scripts/trt_infer.py \
        --engine deploy_onnx/rteir_gopro-720x1280_fp16.engine \
        --height 720 --width 1280 \
        --frames 60 --warmup 20

Benchmark numbers below are host-inclusive (H2D + infer + D2H); see
`scripts/bench_engine.py` for the CUDA-event forward/e2e split.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Try to import tensorrt
try:
    import tensorrt as trt
    TRT_VERSION = trt.__version__
    print(f"TensorRT version: {TRT_VERSION}")
except ImportError:
    print("ERROR: tensorrt not installed. Install with: pip install tensorrt")
    sys.exit(1)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=8, verbose=False):
    """Build TensorRT engine from ONNX model."""
    logger = trt.Logger(trt.Logger.INFO if not verbose else trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled")
        else:
            print("  Warning: FP16 not supported on this platform, using FP32")

    print("Building engine (this may take a few minutes)...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("  Engine build FAILED")
        return None

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(engine)
    print(f"  Engine saved: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    return engine


def load_engine(engine_path):
    """Load serialized TensorRT engine."""
    logger = trt.Logger()
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(f.read())


class TRTRunner:
    """TensorRT inference runner using PyTorch for CUDA memory management."""

    def __init__(self, engine_bytes):
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()

        # Get input/output info
        self.inputs = []
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)
            print(f"  Binding: {name}, mode={mode}, shape={shape}, dtype={dtype}")
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'shape': tuple(shape), 'dtype': dtype})
            else:
                self.outputs.append({'name': name, 'shape': tuple(shape), 'dtype': dtype})

        # Pre-allocate CUDA buffers using PyTorch
        if HAS_TORCH:
            import torch
            self._torch_buffers = {}
            for binding in self.inputs + self.outputs:
                name = binding['name']
                shape = binding['shape']
                torch_dtype = torch.float32 if binding['dtype'] == np.float32 else torch.float16
                buf = torch.empty(shape, dtype=torch_dtype, device='cuda')
                self._torch_buffers[name] = buf
                self.context.set_tensor_address(name, buf.data_ptr())

    def infer(self, input_dict):
        """Run inference. input_dict: {name: numpy array}"""
        if HAS_TORCH:
            import torch
            for name, data in input_dict.items():
                torch_data = torch.from_numpy(data).cuda()
                self._torch_buffers[name].copy_(torch_data)
            self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            results = {}
            for out in self.outputs:
                results[out['name']] = self._torch_buffers[out['name']].cpu().numpy()
            return results
        else:
            raise RuntimeError("PyTorch required for CUDA buffer management")


def benchmark(runner, height, width, img_channels=3, event_channels=6,
              frames=60, warmup=20, dtype=np.float32):
    """Benchmark the TensorRT engine."""
    if HAS_TORCH:
        import torch

    # Detect batch size from engine inputs
    batch_size = runner.inputs[0]['shape'][0] if runner.inputs else 1

    # Create dummy inputs matching engine batch size
    x = np.random.randn(batch_size, img_channels, height, width).astype(dtype)
    event = np.random.randn(batch_size, event_channels, height, width).astype(dtype)

    print(f"\nBenchmarking: {frames} frames, {warmup} warmup, batch={batch_size}, input dtype={dtype.__name__}")
    print(f"  Input: x={x.shape}, event={event.shape}")

    # Warmup
    for _ in range(warmup):
        runner.infer({'x': x, 'event': event})
    if HAS_TORCH:
        import torch
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for i in range(frames):
        start = time.perf_counter()
        runner.infer({'x': x, 'event': event})
        if HAS_TORCH:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times = np.array(times)
    fps = batch_size * 1000 / times.mean()  # images per second
    print(f"\n{'='*60}")
    print(f"Results ({frames} frames, batch={batch_size}):")
    print(f"  Mean:   {times.mean():.2f} ms/batch  ({fps:.2f} img/s)")
    print(f"  Median: {np.median(times):.2f} ms/batch  ({batch_size*1000/np.median(times):.2f} img/s)")
    print(f"  Std:    {times.std():.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")

    if HAS_TORCH:
        import torch
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {peak_mem:.2f} GB")

    return times


def main():
    parser = argparse.ArgumentParser(description='TensorRT inference for RTEIR')
    parser.add_argument('--onnx', type=str, default=None, help='Path to ONNX model')
    parser.add_argument('--engine', type=str, default=None, help='Path to pre-built engine')
    parser.add_argument('--output-engine', type=str, default=None,
                        help='Path to save built engine')
    parser.add_argument('--height', type=int, default=1224, help='Input height')
    parser.add_argument('--width', type=int, default=1632, help='Input width')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    parser.add_argument('--workspace', type=int, default=8, help='Workspace size in GB')
    parser.add_argument('--frames', type=int, default=60, help='Number of frames to benchmark')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup frames')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Build or load engine
    if args.engine:
        print(f"Loading engine: {args.engine}")
        with open(args.engine, 'rb') as f:
            engine_bytes = f.read()
    elif args.onnx:
        engine_path = args.output_engine or args.onnx.replace('.onnx', '_fp16.engine' if args.fp16 else '_fp32.engine')
        engine_bytes = build_engine(args.onnx, engine_path, fp16=args.fp16,
                                     workspace_gb=args.workspace, verbose=args.verbose)
        if engine_bytes is None:
            sys.exit(1)
    else:
        parser.error("Either --onnx or --engine must be specified")

    # Create runner
    print("\nCreating TensorRT runner...")
    runner = TRTRunner(engine_bytes)

    # Benchmark
    dtype = np.float16 if args.fp16 else np.float32

    def _channels(name, default):
        """Read the channel count from the engine binding (6/12/24ch voxels)."""
        for b in runner.inputs:
            shape = b['shape']
            if b['name'] == name and len(shape) == 4 and shape[1] > 0:
                return int(shape[1])
        return default

    benchmark(runner, args.height, args.width,
              img_channels=_channels('x', 3),
              event_channels=_channels('event', 6),
              frames=args.frames,
              warmup=args.warmup, dtype=dtype)


if __name__ == '__main__':
    main()
