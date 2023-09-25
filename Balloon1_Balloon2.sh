#!/usr/bin/env bash
GPUNO=$1
((PORT=$1+29500))
echo $PORT

CUDA_VISIBLE_DEVICES=$GPUNO  python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 \
--master_addr=127.0.0.1 --master_port=$PORT train/train.py \
 -c mononerf_conf/exp/Balloon1_Balloon2/dynamic_temporal_feat_static_mask_flow_loss_multiscenes_discrete2_addition.conf \
  --launcher="pytorch"  