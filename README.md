# Phase Change Analysis

This repository contains code for analyzing phase transitions in neural network training, with specific focus on:
- **Grokking**: Sudden generalization in modular arithmetic tasks
- **Induction Heads**: Emergence of in-context learning in transformers

## Repository Structure

```
phase_change_analysis/
├── grokking/              # Core grokking modules
│   ├── model.py          # Model architectures
│   ├── data.py           # Data generation
│   └── train.py          # Training loop
├── induction_heads/       # Core induction heads modules
│   ├── model.py          # Transformer implementations
│   ├── data.py           # Sequence data generation
│   ├── train.py          # Training loop
│   └── eigen_analysis.py # Eigenvalue analysis
├── utils/                 # Shared utilities
│   ├── analysis.py       # Analysis functions
│   └── visualization.py  # Plotting utilities
├── experiments/           # Experiment scripts
│   ├── grokking/         # Grokking experiments
│   └── induction_heads/  # Induction heads experiments
├── results/              # Saved results and checkpoints
│   ├── grokking/         
│   └── induction_heads/  
└── docs/                 # Documentation

```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Grokking Experiments

```python
# Train a model on modular addition
cd experiments/grokking
python run_experiment.py --task modular_addition --p 97 --steps 50000

# Analyze phase transitions
python analyze_cleanup_phase.py --checkpoint path/to/checkpoint.pt
```

### Induction Heads Experiments

```python
# Train transformer on sequence copying task
cd experiments/induction_heads
python train.py --seq_len 64 --n_heads 2 --steps 10000

# Visualize attention patterns
python visualize_results.py --checkpoint path/to/checkpoint.pt
```

## Key Findings

This codebase implements experiments that demonstrate:

1. **Grokking**: Neural networks can exhibit sudden generalization long after achieving perfect training accuracy
2. **Phase Transitions**: Both phenomena exhibit sharp transitions in their learning dynamics
3. **Eigenvalue Analysis**: Changes in weight matrix eigenvalues correlate with phase transitions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- tqdm
- scipy (for induction heads)
- seaborn (for visualization)

## Citation

If you use this code in your research, please cite:

```bibtex
@repository{phase_change_analysis,
  author = {Edward Turner},
  title = {Phase Change Analysis: Grokking and Induction Heads},
  year = {2025},
  url = {https://github.com/edwardbturner/phase_change_analysis}
}
```