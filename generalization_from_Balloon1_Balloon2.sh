#!/usr/bin/env bash
GPUNO=$1
ITER=$2
((PORT=$1+29600))
echo $PORT


CUDA_VISIBLE_DEVICES=$GPUNO  python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 \
--master_addr=127.0.0.1 --master_port=$PORT train/train_fast_rendering.py \
 -c mononerf_conf/exp/generation_from_Balloon1_Balloon2/Jumping.conf \
  --launcher="pytorch" --ft_path="/data/tianfr/NeRF_series/mononerf/logs/multiscenes/1gpusBalloon1_Balloon2/010000.tar" \
  --fast_render_iter=$2 
