from pathlib import Path

import numpy as np
from torch.utils import data as data

from basicsr.data.event_util import (voxel_norm, voxel_norm_mass,
                                        voxel_norm_perbin,
                                        voxel_norm_gstd, voxel_norm_log)
from basicsr.data.transforms import augment, triple_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor


class goprodataset(data.Dataset):
    """GoPro-style blur/event/gt dataset using precomputed voxel npz."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.phase = str(opt.get("phase", "train")).lower()
        self.scale = int(opt.get("scale", 1))
        self.gt_size = opt.get("gt_size", None)
        self.use_hflip = bool(opt.get("use_hflip", False))
        self.use_rot = bool(opt.get("use_rot", False))
        self.norm_voxel = opt.get("norm_voxel", True)  # bool or 'mass'/'mean'/...
        self.voxel_gstd_sigma = opt.get("voxel_gstd_sigma", 0.9057)
        self.strict_pair = bool(opt.get("strict_pair", True))
        self.voxel_key = str(opt.get("voxel_key", "voxel"))
        self.voxel_auto_transpose = bool(opt.get("voxel_auto_transpose", True))

        self.blur_dir_name = str(opt.get("blur_dir_name", "blur"))
        self.gt_dir_name = str(opt.get("gt_dir_name", "gt"))

        self.dataroot = Path(opt["dataroot"])
        self.dataroot_voxel = Path(opt.get("dataroot_voxel", self.dataroot / "voxel"))

        if not self.dataroot.is_dir():
            raise FileNotFoundError(f"dataroot not found: {self.dataroot}")
        if not self.dataroot_voxel.is_dir():
            raise FileNotFoundError(f"dataroot_voxel not found: {self.dataroot_voxel}")

        self.blur_root = self.dataroot / self.blur_dir_name
        if not self.blur_root.is_dir():
            raise FileNotFoundError(f"blur directory not found: {self.blur_root}")

        self.gt_root = self.dataroot / self.gt_dir_name
        if (not self.gt_root.is_dir()) and self.gt_dir_name == "gt":
            sharp_root = self.dataroot / "sharp"
            if sharp_root.is_dir():
                self.gt_root = sharp_root

        if not self.gt_root.is_dir():
            raise FileNotFoundError(f"gt directory not found: {self.gt_root}")

        self.io_backend_opt = dict(opt["io_backend"])
        self.file_client = None
        self.dataPath = self._build_index()

        logger = get_root_logger()
        logger.info(
            "goprodataset initialized: "
            f"phase={self.phase}, samples={len(self.dataPath)}, "
            f"dataroot={self.dataroot}, dataroot_voxel={self.dataroot_voxel}"
        )

    @staticmethod
    def _list_seq_names(root: Path):
        seq_dirs = sorted([p.name for p in root.iterdir() if p.is_dir()])
        if len(seq_dirs) > 0:
            return seq_dirs
        flat_png = sorted(root.glob("*.png"))
        if len(flat_png) > 0:
            return [""]
        return []

    @staticmethod
    def _list_files(root: Path, seq_name: str, suffix: str):
        seq_root = root if seq_name == "" else root / seq_name
        if not seq_root.is_dir():
            return []
        return sorted([str(p) for p in seq_root.glob(f"*{suffix}")])

    def _build_index(self):
        seq_names = self._list_seq_names(self.blur_root)
        if len(seq_names) == 0:
            raise RuntimeError(f"No samples found under blur root: {self.blur_root}")

        data_list = []
        mismatch_notes = []

        for seq_name in seq_names:
            blur_files = self._list_files(self.blur_root, seq_name, ".png")
            gt_files = self._list_files(self.gt_root, seq_name, ".png")
            voxel_files = self._list_files(self.dataroot_voxel, seq_name, ".npz")

            if self.strict_pair and (
                len(blur_files) != len(gt_files) or len(blur_files) != len(voxel_files)
            ):
                raise ValueError(
                    f"Length mismatch in sequence '{seq_name}': "
                    f"blur={len(blur_files)}, gt={len(gt_files)}, voxel={len(voxel_files)}"
                )

            n = min(len(blur_files), len(gt_files), len(voxel_files))
            if n == 0:
                continue

            if len(blur_files) != len(gt_files) or len(blur_files) != len(voxel_files):
                mismatch_notes.append(
                    f"{seq_name if seq_name else '[flat]'}: "
                    f"blur={len(blur_files)}, gt={len(gt_files)}, voxel={len(voxel_files)}, use={n}"
                )

            for i in range(n):
                stem = Path(blur_files[i]).stem
                image_name = stem if seq_name == "" else f"{seq_name}/{stem}"
                data_list.append(
                    {
                        "blur_path": blur_files[i],
                        "gt_path": gt_files[i],
                        "voxel_path": voxel_files[i],
                        "image_name": image_name,
                    }
                )

        if len(data_list) == 0:
            raise RuntimeError(
                "No valid sample triplets (blur/gt/voxel) were built. "
                "Please check dataroot and dataroot_voxel."
            )

        if len(mismatch_notes) > 0:
            logger = get_root_logger()
            logger.warning(
                "Found length mismatch in some sequences and clipped by min length:\n"
                + "\n".join(mismatch_notes[:20])
            )

        return data_list

    def _init_file_client(self):
        if self.file_client is not None:
            return
        backend = self.io_backend_opt.get("type", "disk")
        backend_opt = {k: v for k, v in self.io_backend_opt.items() if k != "type"}
        self.file_client = FileClient(backend, **backend_opt)

    def _load_voxel_hwc(self, voxel_path: str) -> np.ndarray:
        with np.load(voxel_path) as voxel_npz:
            if self.voxel_key not in voxel_npz.files:
                raise KeyError(
                    f"Voxel key '{self.voxel_key}' not found in {voxel_path}. "
                    f"Available keys: {voxel_npz.files}"
                )
            voxel = np.asarray(voxel_npz[self.voxel_key], dtype=np.float32)

        if voxel.ndim != 3:
            raise ValueError(f"Voxel must be 3D, but got shape {voxel.shape} at {voxel_path}")

        # Accept CHW voxel files and transpose to HWC for cropping/augmentation.
        if self.voxel_auto_transpose and voxel.shape[0] <= 64 and voxel.shape[1] > 64 and voxel.shape[2] > 64:
            voxel = np.transpose(voxel, (1, 2, 0))

        return np.ascontiguousarray(voxel)

    def __getitem__(self, index):
        self._init_file_client()

        item = self.dataPath[index]
        image_path = item["blur_path"]
        gt_path = item["gt_path"]
        voxel_path = item["voxel_path"]

        img_lq = imfrombytes(self.file_client.get(image_path), float32=True)
        img_gt = imfrombytes(self.file_client.get(gt_path), float32=True)
        voxel = self._load_voxel_hwc(voxel_path)

        if self.phase == "train" and self.gt_size is not None:
            img_gt, img_lq, voxel = triple_random_crop(
                img_gt, img_lq, voxel, self.gt_size, self.scale, gt_path
            )

        if self.phase == "train":
            img_lq, img_gt, voxel = augment(
                [img_lq, img_gt, voxel], self.use_hflip, self.use_rot
            )

        img_lq, img_gt, voxel = img2tensor([img_lq, img_gt, voxel])

        if self.norm_voxel == 'mass':
            voxel = voxel_norm_mass(voxel)
        elif self.norm_voxel == 'bin':
            voxel = voxel_norm_perbin(voxel)
        elif self.norm_voxel == 'log':
            voxel = voxel_norm_log(voxel)
        elif self.norm_voxel == 'gstd':
            voxel = voxel_norm_gstd(voxel, sigma=self.voxel_gstd_sigma)
        elif self.norm_voxel:
            voxel = voxel_norm(voxel)

        return {
            "frame": img_lq,
            "frame_gt": img_gt,
            "voxel": voxel,
            "image_name": item["image_name"],
        }

    def __len__(self):
        return len(self.dataPath)


class GoProDataset(goprodataset):
    """Alias class for dataset type name."""

