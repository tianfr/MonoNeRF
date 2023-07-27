# MonoNeRF
This is the official implementation of our paper "MonoNeRF: Learning a Generalizable Dynamic Radiance Field from Monocular Videos".

[![arXiv](https://img.shields.io/badge/arXiv-2212.13056-b31b1b.svg)](https://arxiv.org/abs/2212.13056)

> **MonoNeRF: Learning a Generalizable Dynamic Radiance Field from Monocular Videos**<br>
> [Fengrui Tian](http://tianfr.github.io), [Shaoyi Du](https://gr.xjtu.edu.cn/en/web/dushaoyi/home), [Yueqi Duan](https://duanyueqi.github.io/) <br>
in ICCV 2023 <br>

[arxiv](https://arxiv.org/abs/2212.13056) / [paper](#)

# Introdution

In this paper, we target at the problem of learning a generalizable dynamic radiance field from monocular videos. Different from most existing NeRF methods that are based on multiple views, monocular videos only contain one view at each timestamp, thereby suffering from ambiguity along the view direction in estimating point features and scene flows. Previous studies such as DynNeRF disambiguate point features by positional encoding, which is not transferable and severely limits the generalization ability. As a result, these methods have to train one independent model for each scene and suffer from heavy computational costs when applying to increasing monocular videos in real-world applications. To address this, We propose MonoNeRF to simultaneously learn point features and scene flows with point trajectory and feature correspondence constraints across frames. More specifically, we learn an implicit velocity field to estimate point trajectory from temporal features with Neural ODE, which is followed by a flow-based feature aggregation module to obtain spatial features along the point trajectory. We jointly optimize temporal and spatial features in an end-to-end manner. Experiments show that our MonoNeRF is able to learn from multiple scenes and support new applications such as scene editing, unseen frame synthesis, and fast novel scene adaptation.

![image](https://github.com/tianfr/MonoNeRF/assets/44290909/2f5e9054-f247-4b58-b9b5-7bd2b8e091fd)

**Codes will be released soon. Stay tuned!**


