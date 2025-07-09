import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import create_data_loader, generate_modular_addition_data
from model import ModularAdditionTransformer
from tqdm import tqdm


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(model, data_loader, device):
    """Compute accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_grokking(args):
    """Main training function for grokking experiments."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    set_seed(args.seed)

    # Generate data
    print(f"Generating modular addition data with p={args.p}")
    train_data, test_data = generate_modular_addition_data(p=args.p, train_frac=args.train_frac, seed=args.seed)

    # Create data loaders
    train_loader = create_data_loader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = create_data_loader(test_data, batch_size=args.batch_size, shuffle=False)

    print(f"Train size: {len(train_data[0])}, Test size: {len(test_data[0])}")

    # Initialize model
    model = ModularAdditionTransformer(
        p=args.p, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2)
    )

    # Training history
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # Training loop
    print(f"Starting training for {args.num_steps} steps")
    step = 0
    epoch = 0

    with tqdm(total=args.num_steps) as pbar:
        while step < args.num_steps:
            epoch += 1
            for inputs, labels in train_loader:
                if step >= args.num_steps:
                    break

                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                model.train()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log metrics periodically
                if step % args.log_interval == 0:
                    # Compute accuracies
                    train_acc = compute_accuracy(model, train_loader, device)
                    test_acc = compute_accuracy(model, test_loader, device)

                    # Compute test loss
                    model.eval()
                    test_loss = 0
                    with torch.no_grad():
                        for test_inputs, test_labels in test_loader:
                            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                            test_outputs = model(test_inputs)
                            test_loss += criterion(test_outputs, test_labels).item()
                    test_loss /= len(test_loader)

                    # Store metrics
                    train_losses.append(loss.item())
                    test_losses.append(test_loss)
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)

                    pbar.set_postfix(
                        {
                            "train_loss": f"{loss.item():.4f}",
                            "test_loss": f"{test_loss:.4f}",
                            "train_acc": f"{train_acc:.3f}",
                            "test_acc": f"{test_acc:.3f}",
                        }
                    )

                step += 1
                pbar.update(1)

    # Plot results
    steps = np.arange(len(train_losses)) * args.log_interval

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot losses
    ax1.plot(steps, train_losses, label="Train Loss", alpha=0.8)
    ax1.plot(steps, test_losses, label="Test Loss", alpha=0.8)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Grokking: Loss Curves")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Plot accuracies
    ax2.plot(steps, train_accs, label="Train Accuracy", alpha=0.8)
    ax2.plot(steps, test_accs, label="Test Accuracy", alpha=0.8)
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Grokking: Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"grokking_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")

    # Save final checkpoint if requested
    if args.save_checkpoint:
        checkpoint_path = f"checkpoint_{timestamp}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "train_accs": train_accs,
                "test_accs": test_accs,
                "args": args,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

    # Print final metrics
    print("\nFinal metrics:")
    print(f"Train accuracy: {train_accs[-1]:.3f}")
    print(f"Test accuracy: {test_accs[-1]:.3f}")
    print(f"Train loss: {train_losses[-1]:.4f}")
    print(f"Test loss: {test_losses[-1]:.4f}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Reproduce grokking on modular addition")

    # Data parameters
    parser.add_argument("--p", type=int, default=97, help="Prime modulus")
    parser.add_argument("--train_frac", type=float, default=0.3, help="Fraction of data for training")

    # Model parameters
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # Training parameters
    parser.add_argument("--num_steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1.0, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta2")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_checkpoint", action="store_true", help="Save model checkpoint")

    args = parser.parse_args()

    # Run training
    train_grokking(args)


if __name__ == "__main__":
    main()
