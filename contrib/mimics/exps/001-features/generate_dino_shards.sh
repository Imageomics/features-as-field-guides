#!/usr/bin/env bash

uv run launch.py shards \
    --family dinov3 \
    --ckpt <PATH_TO_CHECKPOINT>/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --shards-root <PATH_TO_SCRATCH_STORAGE>/saev/shards/ \
    --batch-size 128 \
    --n-workers 1 \
    --d-model 1024 \
    --layers 19 21 23 \
    --content-tokens-per-example 384 \
    --max-tokens-per-shard 1200000 \
    data:img-seg-folder  --data.root <PATH_TO_DATASETS>/cambridge-segfolder-v1.6