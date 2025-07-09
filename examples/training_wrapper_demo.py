#!/usr/bin/env python3
"""
Demonstration of the PhaseTransitionTrainer wrapper.

This script shows how to use the training wrapper to monitor phase transitions
during neural network training.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.training import PhaseTransitionTrainer


class SimpleModel(nn.Module):
    """Simple model for demonstration."""

    def __init__(self, input_size=10, hidden_size=50, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_synthetic_data(n_samples=1000, input_size=10, noise_level=0.1):
    """Create synthetic data with a phase transition pattern."""

    # Generate random features
    X = torch.randn(n_samples, input_size)

    # Create a target that shows a phase transition pattern
    # Initially random, then becomes more structured
    y = torch.randint(0, 2, (n_samples,))

    # Add some structure to make it learnable
    # The model should learn to predict based on the first few features
    for i in range(n_samples):
        if X[i, 0] > 0 and X[i, 1] > 0:
            y[i] = 1
        elif X[i, 0] < 0 and X[i, 1] < 0:
            y[i] = 0

    # Add noise
    X += noise_level * torch.randn_like(X)

    return X, y


def main():
    """Main demonstration function."""

    print("=" * 60)
    print("PhaseTransitionTrainer Demonstration")
    print("=" * 60)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data
    print("\nCreating synthetic data...")
    X_train, y_train = create_synthetic_data(n_samples=2000, noise_level=0.05)
    X_val, y_val = create_synthetic_data(n_samples=500, noise_level=0.05)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    model = SimpleModel(input_size=10, hidden_size=50, output_size=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Set up training components
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Create trainer
    trainer = PhaseTransitionTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        log_dir=Path("runs/demo_run"),
        track_gradients=True,
        track_hessian=True,
        hessian_method="diagonal",
        hessian_freq=50,
        checkpoint_freq=100,
        log_freq=10,
        scheduler=scheduler,
        gradient_clip=1.0,
        early_stopping_patience=10,
        early_stopping_min_delta=1e-4,
    )

    # Define a custom callback
    def phase_transition_callback(trainer, epoch, metrics):
        """Callback to monitor for phase transitions."""
        if len(trainer.metrics_history["train_acc"]) > 20:
            recent_acc = trainer.metrics_history["train_acc"][-10:]
            if max(recent_acc) - min(recent_acc) > 0.3:  # 30% jump
                print(f"\n🚀 Potential phase transition detected at epoch {epoch}!")
                print(f"   Recent accuracy range: {min(recent_acc):.3f} - {max(recent_acc):.3f}")

    # Train the model
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, num_epochs=50, callbacks=[phase_transition_callback]
    )

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    summary = trainer.get_training_summary()
    print(f"Final epoch: {summary['final_epoch']}")
    print(f"Total steps: {summary['total_training_steps']}")
    print(f"Final train accuracy: {summary['final_train_acc']:.3f}")
    if summary["final_val_acc"] is not None:
        print(f"Final val accuracy: {summary['final_val_acc']:.3f}")
    else:
        print("Final val accuracy: N/A")
    print(f"Phase transitions detected: {summary['phase_transitions']}")
    print(f"Results saved to: {summary['log_dir']}")

    # Analyze training dynamics
    print("\n" + "-" * 40)
    print("Training Dynamics Analysis")
    print("-" * 40)

    metrics = trainer.metrics_history
    if len(metrics["train_acc"]) > 10:
        # Find the biggest accuracy jump
        acc_diffs = np.diff(metrics["train_acc"])
        max_jump_idx = np.argmax(acc_diffs)
        max_jump = acc_diffs[max_jump_idx]

        print(f"Biggest accuracy jump: {max_jump:.3f} at step {metrics['step'][max_jump_idx]}")

        # Check gradient norm evolution
        if len(metrics["grad_norm"]) > 10:
            early_grad_norm = np.mean(metrics["grad_norm"][:10])
            late_grad_norm = np.mean(metrics["grad_norm"][-10:])
            print(f"Gradient norm evolution: {early_grad_norm:.3f} → {late_grad_norm:.3f}")

    print("\nCheck the log directory for detailed metrics and checkpoints:")
    print(f"  {summary['log_dir']}")
    print("  - metrics.json: Training metrics")
    print("  - best_model.pt: Best model checkpoint")
    print("  - final_model.pt: Final model checkpoint")


if __name__ == "__main__":
    main()
