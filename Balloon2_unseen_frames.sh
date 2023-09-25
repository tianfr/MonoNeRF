#!/usr/bin/env bash
GPUNO=$1
((PORT=$1+29600))
echo $PORT

CUDA_VISIBLE_DEVICES=$GPUNO  python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 \
--master_addr=127.0.0.1 --master_port=$PORT train/train_quarter_frame_training.py \
 -c mononerf_conf/exp/quarter_frame_training/Balloon2.conf \
  --launcher="pytorch"  --N_static_iters=100001