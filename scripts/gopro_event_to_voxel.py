import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


def binary_events_to_voxel_grid(
    events: np.ndarray, num_bins: int, width: int, height: int
) -> np.ndarray:
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"events must be Nx4, but got shape {events.shape}")
    if num_bins <= 0 or width <= 0 or height <= 0:
        raise ValueError("num_bins/width/height must be > 0")

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    delta_t = last_stamp - first_stamp
    if delta_t == 0:
        delta_t = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / delta_t
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int32)
    ys = events[:, 2].astype(np.int32)
    pols = events[:, 3]
    pols[pols == 0] = -1

    tis = ts.astype(np.int32)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_left = tis < num_bins
    np.add.at(
        voxel_grid,
        xs[valid_left] + ys[valid_left] * width + tis[valid_left] * width * height,
        vals_left[valid_left],
    )

    valid_right = (tis + 1) < num_bins
    np.add.at(
        voxel_grid,
        xs[valid_right]
        + ys[valid_right] * width
        + (tis[valid_right] + 1) * width * height,
        vals_right[valid_right],
    )

    return voxel_grid.reshape(num_bins, height, width)


def _resolve_event_keys(keys: Sequence[str]) -> Tuple[str, str, str, str]:
    key_set = set(keys)
    candidates = [
        ("t", "x", "y", "p"),
        ("timestamp", "x", "y", "polarity"),
    ]
    for cand in candidates:
        if set(cand).issubset(key_set):
            return cand
    raise KeyError(
        f"Cannot find event keys in npz. Got keys={sorted(keys)}; expected one of {candidates}"
    )


def load_event_quad(event_path: str) -> np.ndarray:
    with np.load(event_path) as event_npz:
        k_t, k_x, k_y, k_p = _resolve_event_keys(event_npz.files)
        t = np.asarray(event_npz[k_t], dtype=np.float32).reshape(-1)
        x = np.asarray(event_npz[k_x], dtype=np.float32).reshape(-1)
        y = np.asarray(event_npz[k_y], dtype=np.float32).reshape(-1)
        p = np.asarray(event_npz[k_p], dtype=np.float32).reshape(-1)

    if not (len(t) == len(x) == len(y) == len(p)):
        raise ValueError(
            f"Length mismatch in {event_path}: "
            f"len(t)={len(t)}, len(x)={len(x)}, len(y)={len(y)}, len(p)={len(p)}"
        )
    if len(t) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    return np.stack((t, x, y, p), axis=1).astype(np.float32, copy=False)


def infer_hw_from_reference(dataroot: Path, ref_dir_name: str) -> Tuple[int, int]:
    ref_dir = dataroot / ref_dir_name
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")

    first_png = None
    for path in sorted(ref_dir.rglob("*.png")):
        first_png = path
        break
    if first_png is None:
        raise FileNotFoundError(f"No png found under reference directory: {ref_dir}")

    with Image.open(first_png) as img:
        width, height = img.size
    return height, width


def list_event_npz_files(event_root: Path) -> List[Path]:
    return sorted(event_root.rglob("*.npz"))


def _cast_voxel_dtype(voxel: np.ndarray, save_dtype: str) -> np.ndarray:
    if save_dtype == "float16":
        return voxel.astype(np.float16, copy=False)
    if save_dtype == "float32":
        return voxel.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported save_dtype: {save_dtype}")


def _save_voxel_npz(out_path: Path, voxel_hwc: np.ndarray, compressed: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(out_path, voxel=voxel_hwc)
    else:
        np.savez(out_path, voxel=voxel_hwc)


def _event_to_voxel_task(task: Tuple[str, str, int, int, int, str, bool]) -> Tuple[str, str]:
    event_path, out_path, num_bins, height, width, save_dtype, compressed = task
    event_quad = load_event_quad(event_path)

    if len(event_quad) == 0:
        voxel_chw = np.zeros((num_bins, height, width), dtype=np.float32)
    else:
        voxel_chw = binary_events_to_voxel_grid(
            event_quad, num_bins=num_bins, width=width, height=height
        )
    voxel_hwc = np.ascontiguousarray(voxel_chw.transpose(1, 2, 0))
    voxel_hwc = _cast_voxel_dtype(voxel_hwc, save_dtype=save_dtype)

    _save_voxel_npz(Path(out_path), voxel_hwc, compressed=compressed)
    return event_path, "ok"


def _resolve_num_workers(num_workers: int) -> int:
    if num_workers <= 0:
        return max(1, os.cpu_count() or 1)
    return num_workers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline convert event npz to voxel npz for faster dataloader."
    )
    parser.add_argument("--dataroot", type=str, required=True, help="Dataset root path.")
    parser.add_argument(
        "--event-dir-name",
        type=str,
        default="event",
        help="Event directory name under dataroot.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root. Default is dataroot.",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="voxel",
        help="Output directory name under output-root.",
    )
    parser.add_argument(
        "--ref-dir-name",
        type=str,
        default="blur",
        help="Reference image directory for inferring H/W when not provided.",
    )
    parser.add_argument("--height", type=int, default=None, help="Voxel height.")
    parser.add_argument("--width", type=int, default=None, help="Voxel width.")
    parser.add_argument("--num-bins", type=int, default=6, help="Voxel bin count.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="CPU workers. <=0 uses all CPU cores.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only process first N event files for quick check.",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Saved voxel dtype.",
    )
    parser.add_argument(
        "--compressed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use np.savez_compressed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing voxel npz files.",
    )
    return parser.parse_args()


def _iter_tasks(
    event_files: Iterable[Path],
    event_root: Path,
    voxel_root: Path,
    num_bins: int,
    height: int,
    width: int,
    save_dtype: str,
    compressed: bool,
    overwrite: bool,
) -> Tuple[List[Tuple[str, str, int, int, int, str, bool]], int]:
    tasks: List[Tuple[str, str, int, int, int, str, bool]] = []
    skip_exists = 0

    for event_path in event_files:
        rel_path = event_path.relative_to(event_root)
        out_path = voxel_root / rel_path
        if out_path.exists() and not overwrite:
            skip_exists += 1
            continue
        tasks.append(
            (
                str(event_path),
                str(out_path),
                num_bins,
                height,
                width,
                save_dtype,
                compressed,
            )
        )
    return tasks, skip_exists


def main() -> None:
    args = parse_args()

    dataroot = Path(args.dataroot)
    if not dataroot.is_dir():
        raise FileNotFoundError(f"dataroot does not exist: {dataroot}")

    event_root = dataroot / args.event_dir_name
    if not event_root.is_dir():
        raise FileNotFoundError(f"event directory does not exist: {event_root}")

    output_root = Path(args.output_root) if args.output_root else dataroot
    voxel_root = output_root / args.output_dir_name
    voxel_root.mkdir(parents=True, exist_ok=True)

    if args.height is None or args.width is None:
        infer_h, infer_w = infer_hw_from_reference(dataroot, args.ref_dir_name)
        height = infer_h if args.height is None else args.height
        width = infer_w if args.width is None else args.width
    else:
        height = args.height
        width = args.width

    if args.num_bins <= 0:
        raise ValueError("--num-bins must be > 0")
    if height <= 0 or width <= 0:
        raise ValueError("height/width must be > 0")

    event_files = list_event_npz_files(event_root)
    if len(event_files) == 0:
        raise RuntimeError(f"No event npz files found under: {event_root}")
    if args.max_samples is not None:
        event_files = event_files[: args.max_samples]

    tasks, skip_exists = _iter_tasks(
        event_files=event_files,
        event_root=event_root,
        voxel_root=voxel_root,
        num_bins=args.num_bins,
        height=height,
        width=width,
        save_dtype=args.save_dtype,
        compressed=args.compressed,
        overwrite=args.overwrite,
    )

    num_workers = _resolve_num_workers(args.num_workers)
    stats = {"ok": 0, "skip_exists": skip_exists, "error": 0}

    print(f"[INFO] dataroot: {dataroot}")
    print(f"[INFO] event_root: {event_root}")
    print(f"[INFO] voxel_root: {voxel_root}")
    print(f"[INFO] output voxel shape: ({height}, {width}, {args.num_bins})")
    print(f"[INFO] total event files: {len(event_files)}")
    print(f"[INFO] tasks to process: {len(tasks)}")
    print(f"[INFO] skip_exists: {skip_exists}")
    print(f"[INFO] workers: {num_workers}")

    if len(tasks) == 0:
        print("[DONE] Nothing to process.")
        return

    start = time.time()

    if num_workers == 1:
        for idx, task in enumerate(tasks, start=1):
            try:
                _, status = _event_to_voxel_task(task)
                stats[status] = stats.get(status, 0) + 1
            except Exception as ex:
                stats["error"] += 1
                print(f"[ERROR] {task[0]} -> {ex}")
            if idx % 200 == 0 or idx == len(tasks):
                print(
                    f"[PROGRESS] {idx}/{len(tasks)} | ok={stats['ok']} | error={stats['error']}"
                )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_event_to_voxel_task, task) for task in tasks]
            for idx, fut in enumerate(as_completed(futures), start=1):
                try:
                    _, status = fut.result()
                    stats[status] = stats.get(status, 0) + 1
                except Exception as ex:
                    stats["error"] += 1
                    print(f"[ERROR] worker exception: {ex}")

                if idx % 200 == 0 or idx == len(futures):
                    print(
                        f"[PROGRESS] {idx}/{len(futures)} | ok={stats['ok']} | error={stats['error']}"
                    )

    elapsed = time.time() - start
    print("[DONE]")
    print(f"[DONE] stats: {stats}")
    print(f"[DONE] elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
