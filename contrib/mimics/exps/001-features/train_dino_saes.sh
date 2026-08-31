#!/usr/bin/env bash

uv run launch.py train \
    --sweep ~/features-as-field-guides/contrib/mimics/exps/001-features/train_16k_dino.py \
    --max-parallel 4 \
    sae.activation:batch-top-k