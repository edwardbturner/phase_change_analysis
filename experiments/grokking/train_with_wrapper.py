#!/usr/bin/env python3
"""
Train grokking (modular addition) using the unified PhaseTransitionTrainer wrapper.
This script is fully equivalent to the induction heads pipeline in logging, checkpointing, and analysis.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from phase_change_analysis.grokking.data import (
    create_data_loader,
    generate_modular_addition_data,
)
from phase_change_analysis.grokking.model import ModularAdditionTransformer
from phase_change_analysis.utils.training import PhaseTransitionTrainer


def main():
    parser = argparse.ArgumentParser(description="Grokking with PhaseTransitionTrainer")
    parser.add_argument("--p", type=int, default=97, help="Prime modulus")
    parser.add_argument("--train_frac", type=float, default=0.3, help="Fraction of data for training")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["transformer", "mlp"], help="Model type")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--num_steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1.0, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval (steps)")
    parser.add_argument("--hessian_freq", type=int, default=500, help="Hessian logging frequency (steps)")
    parser.add_argument(
        "--hessian_method",
        type=str,
        default="diagonal",
        choices=["diagonal", "eigenvalues"],
        help="Hessian analysis method",
    )
    parser.add_argument("--track_hessian", action="store_true", help="Enable Hessian analysis")
    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Early stopping patience (epochs)")
    parser.add_argument("--gradient_clip", type=float, default=None, help="Gradient clipping value")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory for logs and checkpoints")
    args = parser.parse_args()

    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Data
    train_data, test_data = generate_modular_addition_data(p=args.p, train_frac=args.train_frac, seed=args.seed)
    train_loader = create_data_loader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = create_data_loader(test_data, batch_size=args.batch_size, shuffle=False)

    # Model
    if args.model_type == "transformer":
        model = ModularAdditionTransformer(
            p=args.p,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    else:  # mlp
        from phase_change_analysis.grokking.model import SimpleModularAdditionMLP

        model = SimpleModularAdditionMLP(
            p=args.p,
            hidden_size=args.d_model,
            num_layers=args.n_layers,
        )

    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    loss_fn = nn.CrossEntropyLoss()

    # Scheduler (optional, can be added)
    scheduler = None

    # Trainer
    trainer = PhaseTransitionTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        track_gradients=True,
        track_hessian=args.track_hessian,
        hessian_method=args.hessian_method,
        hessian_freq=args.hessian_freq,
        checkpoint_freq=1000,
        log_freq=args.log_interval,
        scheduler=scheduler,
        gradient_clip=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
    )

    # Calculate number of epochs to reach num_steps
    steps_per_epoch = len(train_loader)
    num_epochs = (args.num_steps + steps_per_epoch - 1) // steps_per_epoch

    print(f"Training for {num_epochs} epochs to reach {args.num_steps} steps...")
    trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=num_epochs,
    )

    # Print summary
    summary = trainer.get_training_summary()
    print("\nFinal summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Save metrics and checkpoints are handled by the trainer


if __name__ == "__main__":
    main()
