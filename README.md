# MonoNeRF
This is the official implementation of our ICCV 2023 paper "MonoNeRF: Learning a Generalizable Dynamic Radiance Field from Monocular Videos".

[![arXiv](https://img.shields.io/badge/arXiv-2212.13056-b31b1b.svg)](https://arxiv.org/abs/2212.13056)

> **MonoNeRF: Learning a Generalizable Dynamic Radiance Field from Monocular Videos**<br>
> [Fengrui Tian](http://tianfr.github.io), [Shaoyi Du](https://gr.xjtu.edu.cn/en/web/dushaoyi/home), [Yueqi Duan](https://duanyueqi.github.io/) <br>
in ICCV 2023 <br>

[arxiv](https://arxiv.org/abs/2212.13056) / [paper](#) / [video](https://youtu.be/A6O4Q3PZZ18)

# Introdution

In this paper, we target at the problem of learning a generalizable dynamic radiance field from monocular videos. Different from most existing NeRF methods that are based on multiple views, monocular videos only contain one view at each timestamp, thereby suffering from ambiguity along the view direction in estimating point features and scene flows. Previous studies such as DynNeRF disambiguate point features by positional encoding, which is not transferable and severely limits the generalization ability. As a result, these methods have to train one independent model for each scene and suffer from heavy computational costs when applying to increasing monocular videos in real-world applications. To address this, We propose MonoNeRF to simultaneously learn point features and scene flows with point trajectory and feature correspondence constraints across frames. More specifically, we learn an implicit velocity field to estimate point trajectory from temporal features with Neural ODE, which is followed by a flow-based feature aggregation module to obtain spatial features along the point trajectory. We jointly optimize temporal and spatial features in an end-to-end manner. Experiments show that our MonoNeRF is able to learn from multiple scenes and support new applications such as scene editing, unseen frame synthesis, and fast novel scene adaptation.

![teaser](https://github.com/tianfr/MonoNeRF/assets/44290909/8308c051-6746-4638-bfe4-558d3c17c7ff)


## Environment Setup
The code is tested with
* Ubuntu 16.04
* Anaconda 3
* Python 3.8.12
* CUDA 11.1
* A100 or 3090 GPUs


To get started, please create the conda environment `mononerf` by running
```
conda create --name mononerf python=3.8
conda activate mononerf

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install imageio==2.19.2 pyhocon==0.3.60  pyparsing==2.4.7 configargparse==1.5.3 tensorboard==2.13.0 ipdb==0.13.13 imgviz==1.7.2 imageio--ffmpeg==0.4.8 
pip install mmcv-full==1.7.1
```
Then install [MMAction2](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) v0.24.1 manually.

```
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
git checkout v0.24.1
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without re-installation.
```
Install the torchdiffeq if you want to use [Neural ODE](https://arxiv.org/abs/1806.07366) for calculating trajectories.
```
pip install torchdiffeq==0.0.1
```
Install other dependencies.
```
pip install tqdm Pillow==9.1.1
```
Finally, clone the MonoNeRF project:
```
git clone https://github.com/tianfr/MonoNeRF.git
cd mononerf
```
## Dynamic Scene Dataset
The [Dynamic Scene Dataset](https://www-users.cse.umn.edu/~jsyoon/dynamic_synth/) is used to
quantitatively evaluate our method. Please refer to the official dataset to download the data. Here we present the data link from [DynamicNeRF](https://github.com/gaochen315/DynamicNeRF).
```
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/data.zip
unzip data.zip
rm data.zip
```
## Backbone Checkpoints
Download the [SlowOnly](https://arxiv.org/abs/1812.03982) pretrained model from [MMAction2](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) website.
```
mkdir checkpoints
wget -P checkpoints/ https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912-3f9ce182.pth
```


## Training
### Environment Initialization
```
export PYTHONPATH=.
```
All the training procedure is conducted on **GPU 0** by default.
### Multiple scenes
You can train a model from scratch by running:
```
chmod +x Balloon1_Balloon2.sh
./Balloon1_Balloon2.sh 0
```

### Unseen frames
Train model for rendering novel views on unseen frames:
```
chmod +x Balloon2_unseen_frames.sh
./Balloon2_unseen_frames.sh 0
```

### Unseen scenes
Test the generalization ability on unseen scenes:
```
chmod +x generalization_from_Balloon1_Balloon2.sh
./generalization_from_Balloon1_Balloon2.sh 0 2000
```
# License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details.

If you find this code useful for your research, please consider citing the following paper:
```
@inproceedings{23iccv/tian_mononerf,
    author    = {Fengrui, Tian and Shaoyi, Du and Yueqi, Duan},
    title     = {{MonoNeRF}: Learning a Generalizable Dynamic Radiance Field from Monocular Videos},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
    year      = {2023}
}
```
# Acknowledgement
Our code is built upon [NeRF](https://github.com/bmild/nerf), [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch), [NSFF](https://github.com/zl548/Neural-Scene-Flow-Fields), [DynamicNeRF](https://github.com/gaochen315/DynamicNeRF), [pixelNeRF](https://github.com/sxyu/pixel-nerf), and [Occupancy Flow](https://github.com/autonomousvision/occupancy_flow). Our flow prediction code is modified from [RAFT](https://github.com/princeton-vl/RAFT). Our depth prediction code is modified from [MiDaS](https://github.com/isl-org/MiDaS).
# Contact
If you have any questions, please feel free to contact [Fengrui Tian](https://tianfr.github.io).
