<div align="center">
<h1>UniSplat: Unified Spatio-Temporal Fusion via 3D Latent Scaffolds for Dynamic Driving Scene Reconstruction</h1>

<a href="https://arxiv.org/html/2511.04595v1"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://chenshi3.github.io/unisplat.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
</div>

**UniSplat** is a feed-forward reconstruction framework in autonomous driving scenarios. Unlike traditional methods that require dense, overlapping views and per-scene optimization, UniSplat achieves state-of-the-art performance through a novel unified 3D latent scaffold representation that seamlessly integrates spatial and temporal information.

https://github.com/user-attachments/assets/5356161a-34b0-4478-977c-4107f24e2750

## Updates
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

## Citation
Please consider citing our work as follows if it is helpful.
```
@article{shi2025unisplat,
  title={UniSplat: Unified Spatio-Temporal Fusion via 3D Latent Scaffolds for Dynamic Driving Scene Reconstruction},
  author={Shi, Chen and Shi, Shaoshuai and Lyu, Xiaoyang and Liu, Chunyang and Sheng, Kehua and Zhang, Bo and Jiang, Li},
  journal={arXiv preprint arXiv:2511.04595},
  year={2025}
}
```

## Acknowledgements

UniSplat uses code from a few open source repositories. Without the efforts of these folks (and their willingness to release their implementations), UniSplat would not be possible. Thanks to these great repositories: [VGGT](https://github.com/facebookresearch/vggt), [MoGe](https://github.com/microsoft/MoGe), [Dino](https://github.com/facebookresearch/dinov2), [Pi3](https://github.com/yyfz/Pi3), [Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs), [Omni-Scene](https://github.com/WU-CVGL/Omni-Scene).


