# Phase Transition Training Wrapper

The `PhaseTransitionTrainer` is a comprehensive training wrapper designed to monitor and analyze phase transitions during neural network training. It provides automatic gradient tracking, Hessian computation, phase transition detection, and comprehensive logging.

## Features

- **Gradient Tracking**: Automatic computation and logging of gradient norms
- **Hessian Analysis**: Optional diagonal or eigenvalue computation of the Hessian matrix
- **Phase Transition Detection**: Automatic detection of sudden changes in training metrics
- **Comprehensive Logging**: Detailed metrics, checkpoints, and JSON exports
- **Early Stopping**: Built-in early stopping with configurable patience
- **Learning Rate Scheduling**: Support for any PyTorch learning rate scheduler
- **Gradient Clipping**: Optional gradient clipping for training stability
- **Callback System**: Custom callbacks for additional monitoring

## Quick Start

```python
from utils.training import PhaseTransitionTrainer
import torch
import torch.nn as nn
import torch.optim as optim

# Your model and data
model = YourModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
train_loader = your_data_loader
val_loader = your_validation_loader

# Create trainer
trainer = PhaseTransitionTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    track_gradients=True,
    track_hessian=True,
    hessian_method="diagonal",
    log_dir="runs/my_experiment",
    checkpoint_freq=100,
    log_freq=10,
)

# Train with automatic monitoring
results = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100
)

# Check for phase transitions
transitions = trainer.detect_phase_transitions()
print(f"Phase transitions detected at steps: {transitions}")
```

## Configuration Options

### Basic Configuration
- `model`: Your PyTorch model
- `optimizer`: PyTorch optimizer
- `loss_fn`: Loss function
- `device`: Training device (auto-detects CUDA)

### Logging Configuration
- `log_dir`: Directory for saving logs and checkpoints
- `checkpoint_freq`: How often to save checkpoints (steps)
- `log_freq`: How often to log metrics (steps)

### Monitoring Configuration
- `track_gradients`: Whether to track gradient norms
- `track_hessian`: Whether to compute Hessian information
- `hessian_method`: "diagonal" or "eigenvalues"
- `hessian_freq`: How often to compute Hessian (steps)

### Training Configuration
- `scheduler`: Learning rate scheduler (optional)
- `gradient_clip`: Gradient clipping value (optional)
- `early_stopping_patience`: Early stopping patience (optional)
- `early_stopping_min_delta`: Minimum improvement for early stopping

## Integration Patterns

### 1. Replace Existing Training Loop

```python
# Instead of your custom training loop
trainer = PhaseTransitionTrainer(model, optimizer, loss_fn)
results = trainer.train(train_loader, val_loader, num_epochs=100)
```

### 2. Use as Monitoring Wrapper

```python
# Keep your existing training loop, add monitoring
trainer = PhaseTransitionTrainer(model, optimizer, loss_fn, track_gradients=True)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your existing training code
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Add monitoring
        if step % trainer.log_freq == 0:
            acc = compute_accuracy(output, target)
            trainer._log_step(step, epoch, loss.item(), acc, grad_norm)
```

### 3. Use as Callback System

```python
def training_callback(trainer, epoch, metrics):
    """Custom callback for additional monitoring."""
    if metrics['train_acc'] > 0.9:
        print(f"High accuracy achieved at epoch {epoch}!")

    # Check for phase transitions
    transitions = trainer.detect_phase_transitions()
    if transitions:
        print(f"Phase transition detected at step {transitions[-1]}")

trainer = PhaseTransitionTrainer(model, optimizer, loss_fn)
results = trainer.train(
    train_loader, val_loader, num_epochs=100,
    callbacks=[training_callback]
)
```

## Output Files

The trainer creates several files in the log directory:

- `metrics.json`: Complete training metrics in JSON format
- `checkpoint_step_*.pt`: Regular checkpoints during training
- `best_model.pt`: Best model based on validation loss
- `final_model.pt`: Final model after training

## Phase Transition Detection

The trainer automatically detects phase transitions by monitoring changes in training metrics:

```python
# Detect transitions in validation accuracy
transitions = trainer.detect_phase_transitions(metric="val_acc", window_size=100)

# Detect transitions in training loss
transitions = trainer.detect_phase_transitions(metric="train_loss", window_size=50)
```

## Hessian Analysis

For deeper analysis, enable Hessian computation:

```python
trainer = PhaseTransitionTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    track_hessian=True,
    hessian_method="diagonal",  # or "eigenvalues"
    hessian_freq=50,  # Compute every 50 steps
)
```

This will track:
- Hessian diagonal statistics (mean, std, max)
- Top eigenvalues (if using "eigenvalues" method)

## Examples

See the `examples/` directory for complete examples:

- `training_wrapper_demo.py`: Basic usage demonstration
- `integration_example.py`: Integration patterns with existing code

## Advanced Usage

### Custom Callbacks

```python
def custom_callback(trainer, epoch, metrics):
    """Custom monitoring logic."""
    # Your custom analysis here
    pass

trainer.train(train_loader, val_loader, num_epochs=100,
              callbacks=[custom_callback])
```

### Loading Checkpoints

```python
# Load a previous training run
checkpoint = trainer.load_checkpoint("runs/previous_run/best_model.pt")
print(f"Loaded checkpoint from step {checkpoint['step']}")
```

### Training Summary

```python
summary = trainer.get_training_summary()
print(f"Final accuracy: {summary['final_train_acc']:.3f}")
print(f"Phase transitions: {summary['phase_transitions']}")
print(f"Total steps: {summary['total_training_steps']}")
```

## Performance Considerations

- **Hessian computation** is computationally expensive. Use `hessian_freq` to control frequency.
- **Gradient tracking** has minimal overhead and is recommended for most use cases.
- **Checkpointing** frequency affects disk usage. Adjust `checkpoint_freq` based on your needs.
- **Early stopping** can significantly reduce training time for some tasks.

## Troubleshooting

### Common Issues

1. **Memory errors with Hessian computation**: Reduce `hessian_freq` or use smaller models
2. **Slow training**: Disable Hessian tracking or reduce logging frequency
3. **No phase transitions detected**: Try different metrics or adjust window size
4. **Checkpoint loading errors**: Ensure checkpoint was created with compatible trainer version

### Debug Mode

For debugging, you can enable verbose logging:

```python
trainer = PhaseTransitionTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    log_freq=1,  # Log every step
    checkpoint_freq=10,  # Checkpoint frequently
)
```