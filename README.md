# RTEIR: Towards Real-Time Event-Guided Image Restoration

Dawei Fan<sup>1,2*</sup>, Peihong Xin<sup>1,2*</sup>, Xiongxin Tang<sup>1,2†</sup>, Fanjiang Xu<sup>1,2</sup>

<sup>1</sup> National Key Laboratory of Space Integrated Information System, Institute of Software, Chinese Academy of Sciences

<sup>2</sup> University of Chinese Academy of Sciences, Beijing, China

## Introduction

Official PyTorch implementation of **RTEIR**, a real-time event-guided image restoration network.

On GoPro at 1280x720, RTEIR runs at 41.59 FPS and 35.72 dB in FP32 on an RTX 6000 Ada, and at 142 FPS with an FP16 TensorRT engine. The same FP16 engine runs at 30.7 FPS on a Jetson AGX Thor edge platform.

## Environment setup

```bash
conda create -n RTEIR python=3.10 -y
conda activate RTEIR
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

For ONNX / TensorRT export and benchmarking, additionally install `onnx onnxsim onnxconverter-common tensorrt`.

## Pretrained Models

Download the pretrained RTEIR checkpoint for GoPro from [Google Drive](https://drive.google.com/file/d/1NBGOy3lLPw66DHGA8cse6__EOhYuL0EZ/view?usp=sharing) and place it at:

```bash
checkpoints/rteir_gopro.pth
```

This path is referenced by `options/test/gopro.yml` (`path.pretrain_network_g`).

## Dataset Preparation & Preprocessing

RTEIR training uses GoPro RGB blur/sharp pairs plus a 6-bin signed event voxel per frame.

Download the ESIM-synthesized GoPro Dataset released by [AHDINet](https://github.com/wyang-vis/AHDINet).

Build the 6-bin signed voxel npz files from the raw event npz files:

```bash
bash scripts/build_gopro_voxel.sh ./datasets/GoPro
```

Expected layout:

```
datasets/GoPro/
├── train/{blur,gt,event,voxel}
└── test/{blur,gt,event,voxel}
```

## Training

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 basicsr/train.py -opt options/train/gopro.yml --launcher pytorch
```

Training auto-resumes from `experiments/gopro/training_states/`. Remove that directory to start from scratch.

## Inferencing

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 basicsr/test.py -opt options/test/gopro.yml --launcher pytorch
```

## Deploying

Export the checkpoint to ONNX:

```bash
python scripts/export_onnx.py --yml options/test/gopro.yml --height 720 --width 1280 --out-dir deploy_onnx --base rteir_gopro-720x1280
```

Build the FP16 TensorRT engine:

```bash
python scripts/trt_build_mixed_fp16.py --onnx deploy_onnx/rteir_gopro-720x1280.onnx --engine deploy_onnx/rteir_gopro-720x1280_fp16.engine --preset normchain
```

## Benchmarking

PyTorch FP32 throughput:

```bash
python basicsr/timer.py --model RTEIR --height 720 --width 1280 --pad-to 16 --dtype float32 --device cuda:0
```

Measure engine latency:

```bash
python scripts/bench_engine.py --engine deploy_onnx/rteir_gopro-720x1280_fp16.engine --h 720 --w 1280 --c 6
```

Evaluate engine accuracy:

```bash
python scripts/trt_psnr_eval.py --dataset gopro --engine deploy_onnx/rteir_gopro-720x1280_fp16.engine --ssim
```

## Acknowledgement

This repository is built on [BasicSR](https://github.com/XPixelGroup/BasicSR), and the GoPro event pipeline follows [AHDINet](https://github.com/wyang-vis/AHDINet) and [EFNet](https://github.com/AHupuJR/EFNet). We thank the authors for releasing their code and data.
