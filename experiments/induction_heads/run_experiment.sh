#!/bin/bash

# Run induction head formation experiment with eigenspectrum tracking

echo "Starting induction head formation experiment..."
echo "This will track how transformers develop pattern-matching abilities"
echo ""

# Default experiment - good for initial exploration
python train.py \
    --vocab_size 50 \
    --seq_length 256 \
    --n_sequences 50000 \
    --n_layers 2 \
    --n_heads 8 \
    --d_model 256 \
    --num_steps 20000 \
    --batch_size 64 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --use_lr_schedule \
    --eigen_interval 100 \
    --log_interval 50 \
    --checkpoint_interval 2000 \
    --track_top_k_eigenvalues 10 \
    --seed 42

echo ""
echo "Training complete! To visualize results:"
echo "python visualize_results.py induction_run_*/"