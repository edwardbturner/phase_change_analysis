#!/bin/bash

# Run grokking experiment with typical hyperparameters from Neel Nanda's work
python train.py \
    --p 97 \
    --train_frac 0.3 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 1 \
    --num_steps 50000 \
    --batch_size 512 \
    --lr 1e-3 \
    --weight_decay 1.0 \
    --log_interval 100 \
    --save_checkpoint