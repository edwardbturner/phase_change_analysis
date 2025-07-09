import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import (
    analyze_pattern_completion_accuracy,
    create_induction_data_loader,
    generate_induction_sequences,
)
from eigen_analysis import HessianEigenAnalyzer
from model import InductionTransformer, visualize_attention_patterns
from tqdm import tqdm


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(model, data_loader, pattern_info, device):
    """Compute loss and accuracy metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (sequences, targets) in enumerate(data_loader):
            sequences, targets = sequences.to(device), targets.to(device)

            outputs = model(sequences)

            # Compute loss
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()

            # Compute accuracy
            predictions = outputs.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.numel()

    # Also compute pattern-specific accuracy
    pattern_acc, random_acc, _ = analyze_pattern_completion_accuracy(model, sequences[:100], pattern_info[:100], device)

    return {
        "loss": total_loss / len(data_loader),
        "accuracy": correct / total,
        "pattern_accuracy": pattern_acc,
        "random_accuracy": random_acc,
    }


def train_induction_heads(args):
    """Main training function for induction head formation study."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    set_seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"induction_run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Generate data
    print("Generating induction sequences...")
    sequences, targets, pattern_info = generate_induction_sequences(
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        n_sequences=args.n_sequences,
        min_pattern_length=args.min_pattern_length,
        max_pattern_length=args.max_pattern_length,
        n_repeats_per_seq=args.n_repeats_per_seq,
        seed=args.seed,
    )

    # Split into train/test
    n_train = int(0.9 * len(sequences))
    train_sequences = sequences[:n_train]
    train_targets = targets[:n_train]
    train_pattern_info = pattern_info[:n_train]

    test_sequences = sequences[n_train:]
    test_targets = targets[n_train:]
    test_pattern_info = pattern_info[n_train:]

    # Create data loaders
    train_loader = create_induction_data_loader(
        train_sequences, train_targets, batch_size=args.batch_size, shuffle=True
    )

    test_loader = create_induction_data_loader(test_sequences, test_targets, batch_size=args.batch_size, shuffle=False)

    print(f"Train sequences: {len(train_sequences)}, Test sequences: {len(test_sequences)}")

    # Initialize model
    model = InductionTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_length,
        dropout=args.dropout,
        use_layernorm=args.use_layernorm,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2)
    )

    # Learning rate scheduler
    if args.use_lr_schedule:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=args.lr * 0.1)

    criterion = nn.CrossEntropyLoss()

    # Initialize Hessian analyzer
    eigen_analyzer = HessianEigenAnalyzer(
        model=model,
        loss_fn=criterion,
        device=device,
        track_top_k=args.track_top_k_eigenvalues,
        use_power_iteration=True,
        track_attention_heads=True,
    )

    # Training history
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "pattern_accuracy": [],
        "random_accuracy": [],
        "learning_rates": [],
        "steps": [],
    }

    # Get a fixed batch for Hessian analysis
    hessian_batch = next(iter(train_loader))

    # Training loop
    print(f"\nStarting training for {args.num_steps} steps...")
    step = 0
    epoch = 0

    with tqdm(total=args.num_steps) as pbar:
        while step < args.num_steps:
            epoch += 1

            for batch_idx, (batch_sequences, batch_targets) in enumerate(train_loader):
                if step >= args.num_steps:
                    break

                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)

                # Forward pass
                model.train()
                outputs = model(batch_sequences)

                # Compute loss
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = batch_targets.view(-1)
                loss = criterion(outputs_flat, targets_flat)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()

                if args.use_lr_schedule:
                    scheduler.step()

                # Logging and analysis
                if step % args.log_interval == 0:
                    # Compute metrics
                    train_metrics = compute_metrics(model, train_loader, train_pattern_info, device)
                    test_metrics = compute_metrics(model, test_loader, test_pattern_info, device)

                    # Track eigenspectrum
                    if step % args.eigen_interval == 0 and step > 0:
                        print(f"\n[Step {step}] Computing eigenspectrum...")
                        eigen_analyzer.track_training_step(hessian_batch, step, test_metrics["pattern_accuracy"])

                    # Update history
                    history["train_loss"].append(train_metrics["loss"])
                    history["train_accuracy"].append(train_metrics["accuracy"])
                    history["test_loss"].append(test_metrics["loss"])
                    history["test_accuracy"].append(test_metrics["accuracy"])
                    history["pattern_accuracy"].append(test_metrics["pattern_accuracy"])
                    history["random_accuracy"].append(test_metrics["random_accuracy"])
                    history["learning_rates"].append(optimizer.param_groups[0]["lr"])
                    history["steps"].append(step)

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "loss": f"{train_metrics['loss']:.4f}",
                            "acc": f"{train_metrics['accuracy']:.3f}",
                            "pattern": f"{test_metrics['pattern_accuracy']:.3f}",
                            "random": f"{test_metrics['random_accuracy']:.3f}",
                        }
                    )

                    # Check for phase transition
                    if len(eigen_analyzer.eigenvalue_history) > 10:
                        transition_step = eigen_analyzer.detect_phase_transition()
                        if transition_step and not hasattr(args, "_transition_detected"):
                            print(f"\n*** Phase transition detected at step {transition_step} ***")
                            args._transition_detected = True

                # Save checkpoint periodically
                if step % args.checkpoint_interval == 0 and step > 0:
                    checkpoint = {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "history": history,
                        "eigen_history": eigen_analyzer.eigenvalue_history,
                        "args": args,
                    }
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
                    torch.save(checkpoint, checkpoint_path)

                step += 1
                pbar.update(1)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_train_metrics = compute_metrics(model, train_loader, train_pattern_info, device)
    final_test_metrics = compute_metrics(model, test_loader, test_pattern_info, device)

    print(f"Train - Loss: {final_train_metrics['loss']:.4f}, Accuracy: {final_train_metrics['accuracy']:.3f}")
    print(f"Test  - Loss: {final_test_metrics['loss']:.4f}, Accuracy: {final_test_metrics['accuracy']:.3f}")
    print(f"Pattern Completion: {final_test_metrics['pattern_accuracy']:.3f}")
    print(f"Random Token Accuracy: {final_test_metrics['random_accuracy']:.3f}")

    # Analyze attention patterns
    print("\n=== Attention Pattern Analysis ===")
    for layer_idx in range(args.n_layers):
        for head_idx in range(min(4, args.n_heads)):  # Analyze first 4 heads
            analysis = visualize_attention_patterns(
                model, test_sequences, test_pattern_info, layer_idx=layer_idx, head_idx=head_idx, seq_idx=0
            )
            if analysis:
                print(
                    f"Layer {layer_idx}, Head {head_idx}: "
                    f"Mean induction score = {analysis['mean_induction_score']:.3f}"
                )

    # Save final results
    final_results = {
        "history": history,
        "eigen_history": eigen_analyzer.eigenvalue_history,
        "final_metrics": {"train": final_train_metrics, "test": final_test_metrics},
        "args": vars(args),
    }

    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
        pickle.dump(final_results, f)

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "model_final.pt"))

    print(f"\nResults saved to {output_dir}/")

    return history, eigen_analyzer


def main():
    parser = argparse.ArgumentParser(description="Study induction head formation via eigenspectrum analysis")

    # Data parameters
    parser.add_argument("--vocab_size", type=int, default=50, help="Vocabulary size")
    parser.add_argument("--seq_length", type=int, default=256, help="Sequence length")
    parser.add_argument("--n_sequences", type=int, default=50000, help="Number of sequences")
    parser.add_argument("--min_pattern_length", type=int, default=2, help="Min pattern length")
    parser.add_argument("--max_pattern_length", type=int, default=10, help="Max pattern length")
    parser.add_argument("--n_repeats_per_seq", type=int, default=5, help="Pattern repeats per sequence")

    # Model parameters
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use_layernorm", action="store_true", default=True, help="Use LayerNorm")

    # Training parameters
    parser.add_argument("--num_steps", type=int, default=20000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--use_lr_schedule", action="store_true", help="Use cosine LR schedule")

    # Analysis parameters
    parser.add_argument("--track_top_k_eigenvalues", type=int, default=10, help="Number of eigenvalues to track")
    parser.add_argument("--eigen_interval", type=int, default=100, help="Steps between eigenvalue computations")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging interval")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Checkpoint interval")

    args = parser.parse_args()

    # Run training
    train_induction_heads(args)


if __name__ == "__main__":
    main()
