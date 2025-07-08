# Induction Head Formation Study - Summary

This folder contains a comprehensive framework for studying how induction heads form in transformers through the lens of gradient/Hessian eigenvalue analysis.

## What We Built

### 1. **Data Generation** (`data.py`)
- Creates sequences with repeated patterns (e.g., "A B C ... A B C")
- Tracks pattern locations for targeted accuracy measurement
- Differentiates between pattern completion and random token prediction

### 2. **Model Architecture** (`model.py`)
- Custom transformer with accessible attention weights
- Hooks for gradient/Hessian computation
- Tools to analyze QK (pattern matching) and OV (copying) circuits

### 3. **Eigenvalue Analysis** (`eigen_analysis.py`)
- Efficient Hessian eigenvalue computation using power iteration
- Tracks gradient-eigenvector alignment
- Monitors attention head circuit formation
- Automatic phase transition detection

### 4. **Training Script** (`train.py`)
- Full training pipeline with eigenspectrum tracking
- Monitors pattern vs random accuracy separately
- Saves comprehensive checkpoints and results
- Detects phase transitions automatically

### 5. **Visualization** (`visualize_results.py`)
- Plots training curves and phase transitions
- Shows eigenvalue evolution over time
- Visualizes attention circuit formation
- Creates summary plots of the phase transition

## Key Insights from Anthropic's Research

1. **Phase Transitions**: Induction heads form abruptly during training, not gradually
2. **In-context Learning**: The phase transition coincides with the emergence of in-context learning
3. **Circuit Formation**: QK circuits learn to match patterns, OV circuits learn to copy
4. **Eigenvalue Dynamics**: The phase transition is observable through Hessian eigenvalue changes

## Running Experiments

### Quick Test:
```bash
python test_setup.py
```

### Full Experiment:
```bash
./run_experiment.sh
# or with custom parameters:
python train.py --num_steps 30000 --eigen_interval 100
```

### Visualize Results:
```bash
python visualize_results.py induction_run_*/
```

## What to Look For

1. **Sudden Jump**: Pattern accuracy suddenly increases from ~random to much higher
2. **Eigenvalue Spike**: Top eigenvalues increase rapidly during the transition
3. **Gradient Alignment**: The gradient aligns more with top eigenvectors
4. **Circuit Formation**: QK and OV eigenvalues show distinctive patterns

## GPU Considerations

With an H100:
- Can track more eigenvalues (10-20 instead of 3-5)
- Can use larger models to see clearer transitions
- Eigenvalue computation is fast enough for frequent tracking

## Future Directions

1. Study how architectural choices affect phase transition timing
2. Analyze multiple phase transitions in deeper models
3. Connect eigenvalue dynamics to mechanistic interpretability
4. Explore interventions that accelerate/delay induction head formation

The code is designed to be modular and extensible for further research into neural network phase transitions and emergent capabilities.