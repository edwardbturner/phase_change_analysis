#!/bin/bash

echo "Running complete induction head formation experiment..."
echo "This will take approximately 15-30 minutes on H100"
echo ""

# Run the full experiment with optimized parameters
python train.py \
    --vocab_size 50 \
    --seq_length 256 \
    --n_sequences 50000 \
    --min_pattern_length 2 \
    --max_pattern_length 10 \
    --n_repeats_per_seq 5 \
    --d_model 256 \
    --n_heads 8 \
    --n_layers 2 \
    --d_ff 1024 \
    --dropout 0.1 \
    --use_layernorm \
    --num_steps 20000 \
    --batch_size 64 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.999 \
    --grad_clip 1.0 \
    --use_lr_schedule \
    --track_top_k_eigenvalues 10 \
    --eigen_interval 100 \
    --log_interval 50 \
    --checkpoint_interval 2000 \
    --seed 42

echo ""
echo "Training complete!"
echo ""
echo "To visualize the results, run:"
echo "python visualize_results.py induction_run_*/"