<div align="center">
<h1>UniSplat: Unified Spatio-Temporal Fusion via 3D Latent Scaffolds for Dynamic Driving Scene Reconstruction</h1>

<a href="https://arxiv.org/html/2511.04595v1"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://chenshi3.github.io/unisplat.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
</div>

**UniSplat** is a feed-forward reconstruction framework in autonomous driving scenarios. Unlike traditional methods that require dense, overlapping views and per-scene optimization, UniSplat achieves state-of-the-art performance through a novel unified 3D latent scaffold representation that seamlessly integrates spatial and temporal information.

https://github.com/user-attachments/assets/1214b82a-3a5a-4d65-844a-878f0dbab004

## Updates
- [May 22, 2026] Training code and data preprocessing tools are released.
- [Mar 5, 2026] 🔥 UniSplat is accepted at [ICLR2026](https://openreview.net/forum?id=Ng2VDbKD4r).
- [Dec 7, 2025] Demo code and pretrained weights for the Waymo Dataset have been released. Demo for novel view synthesis (rotation and shift) and scene completion will be released soon.


## Getting Started
### Installation
First, clone this repository and install the dependencies. 
```bash
git clone git@github.com:chenshi3/UniSplat.git
cd UniSplat
pip install -r requirements.txt

## install 3DGS rasterizer
pip install -e submodules/diff-gaussian-rasterization-feature
pip install -e submodules/simple-knn-v2

```

Then download the pretrained model [weights](https://huggingface.co/chenchenshi/UniSplat/blob/main/model.safetensors) and example [data](https://huggingface.co/chenchenshi/UniSplat/blob/main/data.zip) from Hugging Face.

### Quick Demo
Run UniSplat on the provided example data to test the model:
```bash
python demo.py --load_from /path/to/checkpoint.pth --data_path /path/to/demo_data
```
**Arguments:**
- `--load_from`: Path to the pretrained model checkpoint
- `--data_path`: Path to the directory containing example data

The script will process the input data and save the rendered images along with dynamic masks to the output directory.

## Data Preparation

Training and full-scene evaluation use the Waymo Open Perception dataset
(v1.4.3). We keep Waymo's `training/` and `validation/` split: run
the preprocessing steps below twice and dump each split into its own
`scene_root` (e.g. `data/waymo/train/` and `data/waymo/val/`). `train.py`
then points `--data_path` at the train one, `demo.py` at the val one. 

```
{scene_root}/
└── {scene_name}/                                   # e.g. segment-xxxxx_..._with_camera_labels.tfrecord
    ├── images/
    │   └── {frame:06d}_{cam-1}.png                 # frame 6-digit zero-padded, cam 0-indexed
    ├── {frame:05d}_{cam}.exr                       # LiDAR-projected sparse depth, float32
    ├── {frame:05d}_{cam}.npz                       # intrinsics / cam2world / cam2lidar / distortion
    ├── {frame:05d}_{cam}_moge_mask.png             # sky mask, 255 = non-sky, 0 = sky
    └── dynamics/
        ├── dynamic_infos.json                       # per-track speed & metadata
        └── dynamic_mask_{frame_int}_{cam}.npz       # instance id per pixel, uint16
```
### Step 1 — Download Waymo Perception v1.4.x

Accept the [Waymo Open Dataset](https://waymo.com/open/) license and place
the `*.tfrecord` files for the splits you care about in a single directory.

### Step 2 — Extract RGB images
Decode each frame's images out of the tfrecords. We do not ship a dedicated
script for this step because the convention is well covered upstream —
follow either [CUT3R](https://github.com/CUT3R/CUT3R/)
or [Street Crafter](https://github.com/zju3dv/street_crafter)'s
preprocessing. The expected output is `images/{frame:06d}_{cam-1}.png` under
each scene directory (cam-1 ∈ {0,1,2,3,4}).

### Step 3 — Generate depth + calibration

```bash
python tools/preprocess_waymo.py \
    --waymo_dir /path/to/waymo/tfrecords \
    --output_dir /path/to/scene_root \
    --workers 8
```

This single pass produces both `{frame:05d}_{cam}.exr` (LiDAR-projected
sparse depth) and `{frame:05d}_{cam}.npz` (intrinsics + cam2world +
cam2lidar + distortion) for every scene/frame/camera.

Requires the `waymo-open-dataset-tf-2-12-0` package:
```
pip install waymo-open-dataset-tf-2-12-0==1.6.4
```

### Step 4 — Dynamic mask & track info (Coming soon)
We will release the pre-computed dynamic masks and `dynamic_infos.json`
files. Once released, extract it
into your `{scene_root}` so each scene gains a `dynamics/` sub-directory.
### Step 5 — Sky mask

Download `skyseg.onnx` from
[Sky-Segmentation-and-Post-processing](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing),
then run:

```bash
python tools/run_sky_mask.py \
    --scene_root /path/to/scene_root \
    --onnx /path/to/skyseg.onnx \
    --gpus 0,1,2,3 --workers_per_gpu 2
```

Writes `{frame:05d}_{cam}_moge_mask.png` next to the `.exr`/`.npz` files
under each scene.

## Training

Training is done in three stages, each driven by its own config under
`configs/`. The full chain is:

| Stage | Config | Trainable parts |
|---|---|---|
| 1 | `configs/waymo_stage1.yaml` | scale_head / shift_head / point_decoder |
| 2 | `configs/waymo_stage2.yaml` | the rest of `gaussian_head` |
| 3 | `configs/waymo_stage3.yaml` | the rest of `gaussian_head` |

Stage 1 trains the depth-scale/shift heads to match the LiDAR alignment.
Stage 2 freezes those heads and learns the gaussian head against
GT-aligned depths. Stage 3 keeps the heads frozen but swaps in the
predicted scale/shift so the gaussian head adapts to the inference-time
depth distribution.

Launch (example, 8-GPU torchrun):

```bash
# Stage 1
torchrun --nproc_per_node=8 train.py \
    --config configs/waymo_stage1.yaml \
    --data_path /path/to/scene_root \
    --pi3_ckpt /path/to/pi3.safetensors \
    --dinov2_ckpt /path/to/dinov2_vits14_reg4_pretrain.pth

# Stage 2 — set Model.pretrained inside the yaml to point at the stage-1 ckpt
torchrun --nproc_per_node=8 train.py \
    --config configs/waymo_stage2.yaml \
    --data_path /path/to/scene_root

# Stage 3 — set Model.pretrained inside the yaml to point at the stage-2 ckpt
torchrun --nproc_per_node=8 train.py \
    --config configs/waymo_stage3.yaml \
    --data_path /path/to/scene_root
```

Checkpoints land in `./work_dirs/{config_stem}/model_epoch_{N}/`. Resuming
is automatic: the script picks up the latest `model_epoch_*` it finds in
the work dir.

## Citation
Please consider citing our work as follows if it is helpful.
```
@inproceedings{
  shi2026unisplat,
  title={UniSplat: Unified Spatio-Temporal Fusion via 3D Latent Scaffolds for Dynamic Driving Scene Reconstruction},
  author={Chen Shi and Shaoshuai Shi and Xiaoyang Lyu and Chunyang Liu and Kehua Sheng and Bo Zhang and Li Jiang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=Ng2VDbKD4r}
}
```

## Acknowledgements

UniSplat uses code from a few open source repositories. Without the efforts of these folks (and their willingness to release their implementations), UniSplat would not be possible. Thanks to these great repositories: [VGGT](https://github.com/facebookresearch/vggt), [MoGe](https://github.com/microsoft/MoGe), [Dino](https://github.com/facebookresearch/dinov2), [Pi3](https://github.com/yyfz/Pi3), [Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs), [Omni-Scene](https://github.com/WU-CVGL/Omni-Scene).


