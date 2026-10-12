"""Build a TensorRT fp16 engine.

Usage:
    python scripts/trt_build_mixed_fp16.py \
        --onnx <fp32.onnx> --engine <out.engine> --preset normchain
"""

import argparse

import tensorrt as trt

SHAPE_OPS = {'Constant', 'Shape', 'Gather', 'Unsqueeze', 'Squeeze', 'Slice',
             'Concat', 'Reshape', 'Transpose', 'Expand', 'Tile', 'Range', 'Identity'}
STOP_OPS = {'Conv', 'MatMul', 'Gemm', 'ConvTranspose'}


def build_normchain_names(onnx_path, up_hops=3, down_hops=3):
    """Names of the nodes forming the (hand-written) normalization chains.

    Seeds are the reduce/normalization ops; the walk follows producers upstream
    and consumers downstream, stopping at shape plumbing and at conv/matmul
    (which are fine in fp16 and must not be dragged in).
    """
    import onnx
    m = onnx.load(onnx_path)
    name2node = {n.name: n for n in m.graph.node}
    prod = {}
    for n in m.graph.node:
        for o in n.output:
            if o:
                prod[o] = n
    cons = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    seeds = [n for n in m.graph.node
             if n.op_type.startswith('Reduce') or n.op_type in ('LayerNormalization', 'InstanceNormalization')]
    sel = {n.name for n in seeds}

    frontier = list(seeds)
    for _ in range(up_hops):
        nxt = []
        for n in frontier:
            for i in n.input:
                p = prod.get(i)
                if p is None or p.op_type in SHAPE_OPS or p.op_type in STOP_OPS or p.name in sel:
                    continue
                sel.add(p.name)
                nxt.append(p)
        frontier = nxt

    frontier = list(seeds)
    for _ in range(down_hops):
        nxt = []
        for n in frontier:
            for o in n.output:
                for c in cons.get(o, []):
                    if c.op_type in SHAPE_OPS or c.name in sel:
                        continue
                    sel.add(c.name)
                    nxt.append(c)
        frontier = nxt
    return sel


def pin(layer, verbose=False):
    layer.precision = trt.float32
    for j in range(layer.num_outputs):
        if layer.get_output(j).dtype == trt.float16:
            layer.set_output_type(j, trt.float32)
    if verbose:
        print(f'  pin fp32: {layer.name} ({layer.type})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--engine', required=True)
    ap.add_argument('--workspace', type=int, default=8, help='GB')
    ap.add_argument('--preset', default='norm',
                    choices=['norm', 'norm+softmax', 'reduce', 'reduce+unary', 'normchain', 'full'])
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit('ONNX parse failed')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace * (1 << 30))
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    pin_types = set()
    if args.preset in ('norm+softmax', 'full'):
        pin_types.add(trt.LayerType.SOFTMAX)
    if args.preset in ('reduce', 'reduce+unary', 'full'):
        pin_types.add(trt.LayerType.REDUCE)
    if args.preset in ('reduce+unary', 'full'):
        pin_types.add(trt.LayerType.UNARY)
    if args.preset == 'full':
        pin_types.add(trt.LayerType.ELEMENTWISE)

    normchain = build_normchain_names(args.onnx) if args.preset == 'normchain' else set()
    if args.preset == 'normchain':
        print(f'normchain: {len(normchain)} ONNX nodes selected')

    n_pin = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        is_norm = layer.type == trt.LayerType.NORMALIZATION
        is_norm_chain = ('/norm/' in layer.name) or layer.name.endswith('/norm')
        if (is_norm or is_norm_chain or layer.type in pin_types
                or layer.name in normchain):
            pin(layer, verbose=args.verbose)
            n_pin += 1
    print(f'pinned {n_pin} layers to fp32 (preset={args.preset})')

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit('engine build failed')
    with open(args.engine, 'wb') as f:
        f.write(serialized)
    print(f'engine saved: {args.engine}')


if __name__ == '__main__':
    main()
