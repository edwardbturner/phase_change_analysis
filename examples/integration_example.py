#!/usr/bin/env python3
"""
Integration example showing how to use PhaseTransitionTrainer with existing code.

This demonstrates how to wrap existing training loops with the phase transition
monitoring capabilities.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.analysis import compute_gradient_norm
from utils.training import PhaseTransitionTrainer


def integrate_with_existing_training():
    """Example of integrating PhaseTransitionTrainer with existing training code."""

    print("=" * 60)
    print("PhaseTransitionTrainer Integration Example")
    print("=" * 60)

    # Example: You have an existing model and training setup
    class ExistingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(20, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 10)
            )

        def forward(self, x):
            return self.layers(x)

    # Existing training setup
    model = ExistingModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Create synthetic data
    X = torch.randn(1000, 20)
    y = torch.randint(0, 10, (1000,))
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Option 1: Replace existing training loop with trainer
    print("\nOption 1: Replace existing training loop")
    print("-" * 40)

    trainer = PhaseTransitionTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        track_gradients=True,
        track_hessian=True,
        hessian_method="diagonal",
        log_dir=Path("runs/integration_example"),
        checkpoint_freq=50,
        log_freq=5,
    )

    # Use the trainer's main training loop
    trainer.train(train_loader=dataloader, num_epochs=10)

    print("Training completed. Final metrics:")
    summary = trainer.get_training_summary()
    print(f"  Final accuracy: {summary['final_train_acc']:.3f}")
    print(f"  Phase transitions: {summary['phase_transitions']}")

    # Option 2: Use trainer for monitoring while keeping existing loop
    print("\nOption 2: Use trainer for monitoring with existing loop")
    print("-" * 40)

    # Reset model
    model = ExistingModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create trainer for monitoring
    monitor_trainer = PhaseTransitionTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        track_gradients=True,
        track_hessian=False,  # Disable for faster training
        log_dir=Path("runs/monitoring_example"),
        log_freq=10,
    )

    # Your existing training loop
    print("Running existing training loop with monitoring...")
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(dataloader):
            step = epoch * len(dataloader) + batch_idx

            # Your existing training code
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            # Use trainer for logging and monitoring
            if step % monitor_trainer.log_freq == 0:
                # Compute accuracy
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(target.view_as(pred)).sum().item() / target.size(0)

                # Log through trainer
                monitor_trainer._log_step(
                    step,
                    epoch,
                    loss.item(),
                    acc,
                    compute_gradient_norm(model) if monitor_trainer.track_gradients else 0.0,
                )

        # Epoch summary
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Check for phase transitions
    transitions = monitor_trainer.detect_phase_transitions()
    if transitions:
        print(f"Phase transitions detected at steps: {transitions}")

    # Option 3: Use trainer as a callback in existing training
    print("\nOption 3: Use trainer as callback in existing training")
    print("-" * 40)

    # Reset model
    model = ExistingModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create trainer
    callback_trainer = PhaseTransitionTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        track_gradients=True,
        log_dir=Path("runs/callback_example"),
        log_freq=5,
    )

    # Custom training loop with trainer as callback
    def training_step_callback(step, epoch, loss, acc):
        """Callback to integrate trainer monitoring."""
        callback_trainer._log_step(
            step, epoch, loss, acc, compute_gradient_norm(model) if callback_trainer.track_gradients else 0.0
        )

        # Check for phase transitions periodically
        if step % 50 == 0:
            transitions = callback_trainer.detect_phase_transitions()
            if transitions:
                print(f"🚀 Phase transition detected at step {step}!")

    # Your existing training loop
    print("Running training with callback integration...")
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(dataloader):
            step = epoch * len(dataloader) + batch_idx

            # Training step
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).sum().item() / target.size(0)

            # Call callback
            training_step_callback(step, epoch, loss.item(), acc)

    print("Integration examples completed!")
    print("\nKey benefits of using PhaseTransitionTrainer:")
    print("1. Automatic gradient and Hessian tracking")
    print("2. Phase transition detection")
    print("3. Comprehensive logging and checkpointing")
    print("4. Easy integration with existing code")
    print("5. Built-in early stopping and learning rate scheduling")


if __name__ == "__main__":
    integrate_with_existing_training()
