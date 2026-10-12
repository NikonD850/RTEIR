"""Measure engine latency.

Usage:
    python scripts/bench_engine.py --engine <path.engine> --h 720 --w 1280 --c 6
"""

import argparse

import numpy as np
import torch
import tensorrt as trt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--engine', required=True)
    ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--w', type=int, required=True)
    ap.add_argument('--c', type=int, default=6, help='event channels')
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=20)
    args = ap.parse_args()

    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    eng = rt.deserialize_cuda_engine(open(args.engine, 'rb').read())
    if eng is None:
        raise SystemExit('failed to deserialize engine')
    ctx = eng.create_execution_context()
    bufs = {}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        bufs[n] = torch.empty(tuple(eng.get_tensor_shape(n)), dtype=torch.float32, device='cuda')
        ctx.set_tensor_address(n, bufs[n].data_ptr())

    s = torch.cuda.Stream()
    x = torch.rand(1, 3, args.h, args.w, device='cuda')
    ev = torch.randn(1, args.c, args.h, args.w, device='cuda')

    def fwd():
        bufs['x'].copy_(x)
        bufs['event'].copy_(ev)
        ctx.execute_async_v3(s.cuda_stream)

    with torch.cuda.stream(s):
        for _ in range(args.warmup):
            fwd()
    s.synchronize()

    def timeit(fn, n):
        st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        with torch.cuda.stream(s):
            for i in range(n):
                st[i].record(s)
                fn()
                en[i].record(s)
        s.synchronize()
        return np.array([a.elapsed_time(b) for a, b in zip(st, en)])

    # forward-only (buffers already filled)
    with torch.cuda.stream(s):
        bufs['x'].copy_(x)
        bufs['event'].copy_(ev)
    s.synchronize()
    t_fwd = timeit(lambda: ctx.execute_async_v3(s.cuda_stream), args.iters)
    t_e2e = timeit(fwd, args.iters)

    name = args.engine.split('/')[-1]
    print(f'{name}')
    print(f'  forward: {t_fwd.mean():.2f} ms  (median {np.median(t_fwd):.2f}, min {t_fwd.min():.2f})'
          f'  -> {1000 / t_fwd.mean():.1f} img/s')
    print(f'  e2e    : {t_e2e.mean():.2f} ms  (median {np.median(t_e2e):.2f}, min {t_e2e.min():.2f})'
          f'  -> {1000 / t_e2e.mean():.1f} img/s')


if __name__ == '__main__':
    main()
