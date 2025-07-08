import torch
import numpy as np


def generate_modular_addition_data(p, train_frac=0.3, seed=42):
    """
    Generate data for modular addition task (a + b) mod p.

    Args:
        p: Prime modulus
        train_frac: Fraction of data to use for training
        seed: Random seed for reproducibility

    Returns:
        train_data, test_data: tuples of (inputs, labels)
    """
    np.random.seed(seed)

    # Generate all possible pairs (a, b) where a, b in [0, p-1]
    all_pairs = []
    all_labels = []

    for a in range(p):
        for b in range(p):
            all_pairs.append([a, b])
            all_labels.append((a + b) % p)

    all_pairs = np.array(all_pairs)
    all_labels = np.array(all_labels)

    # Shuffle data
    n_total = len(all_pairs)
    indices = np.random.permutation(n_total)
    all_pairs = all_pairs[indices]
    all_labels = all_labels[indices]

    # Split into train and test
    n_train = int(n_total * train_frac)

    train_pairs = torch.tensor(all_pairs[:n_train], dtype=torch.long)
    train_labels = torch.tensor(all_labels[:n_train], dtype=torch.long)

    test_pairs = torch.tensor(all_pairs[n_train:], dtype=torch.long)
    test_labels = torch.tensor(all_labels[n_train:], dtype=torch.long)

    return (train_pairs, train_labels), (test_pairs, test_labels)


def create_data_loader(data, batch_size, shuffle=True):
    """Create DataLoader from data tuple."""
    inputs, labels = data
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
