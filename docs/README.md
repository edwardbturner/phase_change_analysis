# Induction Head Formation Analysis via Eigenspectrum

This project studies how induction heads form in transformers through the lens of gradient/Hessian analysis and eigenvalue evolution. It's based on Anthropic's groundbreaking research on induction heads and phase transitions in transformer training.

## Background

Induction heads are attention heads that enable transformers to perform pattern completion - they look for sequences like "A B ... A" and predict that "B" will follow. According to Anthropic's research, these heads form through an abrupt phase transition early in training, coinciding with the emergence of in-context learning abilities.

## Key Features

1. **Data Generation**: Creates sequences with repeated patterns to encourage induction head formation
2. **Eigenspectrum Tracking**: Monitors Hessian eigenvalues/eigenvectors during training
3. **Phase Transition Detection**: Automatically detects when induction heads form
4. **Attention Analysis**: Tracks QK and OV circuit formation
5. **Comprehensive Visualization**: Plots showing the phase transition and eigenvalue dynamics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python train.py
```

### With Custom Parameters

```bash
python train.py \
    --vocab_size 100 \
    --seq_length 512 \
    --n_layers 4 \
    --n_heads 8 \
    --num_steps 30000 \
    --eigen_interval 100
```

### Visualize Results

After training completes:

```bash
python visualize_results.py induction_run_[timestamp]/
```

## Key Parameters

- `--vocab_size`: Size of token vocabulary (default: 50)
- `--seq_length`: Length of input sequences (default: 256)
- `--n_layers`: Number of transformer layers (default: 2)
- `--n_heads`: Number of attention heads per layer (default: 8)
- `--num_steps`: Total training steps (default: 20000)
- `--eigen_interval`: Steps between eigenvalue computations (default: 100)
- `--batch_size`: Training batch size (default: 64)

## Understanding the Results

### 1. Phase Transition
The most important phenomenon to observe is the phase transition where:
- Pattern completion accuracy suddenly jumps from near-random to high accuracy
- This transition is typically abrupt, occurring within a narrow window
- The timing depends on model architecture and training dynamics

### 2. Eigenvalue Evolution
- **Top eigenvalues** increase rapidly during the phase transition
- **Spectral norm** (largest eigenvalue) shows distinctive behavior
- **Gradient alignment** with top eigenvector changes during transition

### 3. Attention Metrics
- **QK eigenvalues**: Indicate formation of pattern-matching circuits
- **OV copy ratio**: Shows when heads learn to copy information

## Output Files

Each training run creates a timestamped directory containing:
- `config.json`: Training configuration
- `results.pkl`: Complete training history and analysis
- `checkpoint_step_*.pt`: Model checkpoints
- `model_final.pt`: Final trained model
- Visualization plots (after running `visualize_results.py`)

## Theory

The project is based on several key insights:

1. **Induction heads** are a specific type of attention pattern that enables in-context learning
2. **Phase transitions** in neural networks can be tracked through eigenspectrum analysis
3. **Hessian eigenvalues** provide information about the loss landscape curvature
4. **Gradient-eigenvector alignment** indicates optimization dynamics

## GPU Requirements

This code is optimized for GPU training. With an H100:
- Full eigenspectrum analysis is feasible even for larger models
- Training typically completes in 10-30 minutes depending on configuration
- Larger models (more layers/heads) will show more complex phase transitions

## References

- [Anthropic: In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

## Troubleshooting

1. **Out of memory**: Reduce `batch_size` or `track_top_k_eigenvalues`
2. **No phase transition observed**: Try longer training (`--num_steps 50000`)
3. **Eigenvalue computation too slow**: Increase `--eigen_interval`