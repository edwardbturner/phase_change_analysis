import random

import numpy as np
import torch


def generate_induction_sequences(
    vocab_size=50,
    seq_length=256,
    n_sequences=10000,
    min_pattern_length=2,
    max_pattern_length=10,
    n_repeats_per_seq=5,
    seed=42,
):
    """
    Generate sequences with repeated patterns to encourage induction head formation.

    Creates sequences like: [random tokens] A B C [random tokens] A B C [random tokens]
    where the model should learn to predict B after seeing A in the second occurrence,
    C after seeing A B, etc.

    Args:
        vocab_size: Size of vocabulary (excluding special tokens)
        seq_length: Length of each sequence
        n_sequences: Number of sequences to generate
        min_pattern_length: Minimum length of repeated patterns
        max_pattern_length: Maximum length of repeated patterns
        n_repeats_per_seq: Number of times to repeat patterns in each sequence
        seed: Random seed

    Returns:
        sequences: torch.LongTensor of shape (n_sequences, seq_length)
        targets: torch.LongTensor of shape (n_sequences, seq_length)
        pattern_info: List of dicts with pattern locations and lengths
    """
    np.random.seed(seed)
    random.seed(seed)

    # Reserve token 0 for padding, actual vocab is 1 to vocab_size
    sequences = []
    targets = []
    pattern_info = []

    for i in range(n_sequences):
        seq = np.zeros(seq_length, dtype=np.int64)
        target = np.zeros(seq_length, dtype=np.int64)
        patterns_in_seq = []

        # Fill with random tokens initially
        seq[:] = np.random.randint(1, vocab_size + 1, seq_length)

        # Add repeated patterns
        used_positions = set()

        for _ in range(n_repeats_per_seq):
            pattern_length = np.random.randint(min_pattern_length, max_pattern_length + 1)

            # Generate a random pattern
            pattern = np.random.randint(1, vocab_size + 1, pattern_length)

            # Find positions to place the pattern (at least 2 occurrences)
            n_occurrences = np.random.randint(2, min(5, seq_length // (pattern_length + 5)) + 1)

            positions = []
            attempts = 0
            while len(positions) < n_occurrences and attempts < 100:
                pos = np.random.randint(0, seq_length - pattern_length + 1)

                # Check if position overlaps with existing patterns
                if not any(pos <= p < pos + pattern_length or p <= pos < p + pattern_length for p in used_positions):
                    positions.append(pos)
                    for j in range(pattern_length):
                        used_positions.add(pos + j)
                attempts += 1

            if len(positions) >= 2:
                # Place the pattern
                for pos in positions:
                    seq[pos : pos + pattern_length] = pattern

                # Record pattern info
                patterns_in_seq.append({"pattern": pattern.tolist(), "positions": positions, "length": pattern_length})

        # Create targets (next token prediction)
        target[:-1] = seq[1:]
        target[-1] = 0  # Padding for last position

        sequences.append(seq)
        targets.append(target)
        pattern_info.append(patterns_in_seq)

    return (
        torch.tensor(np.array(sequences), dtype=torch.long),
        torch.tensor(np.array(targets), dtype=torch.long),
        pattern_info,
    )


def create_induction_data_loader(sequences, targets, batch_size, shuffle=True):
    """Create DataLoader for induction head training."""
    dataset = torch.utils.data.TensorDataset(sequences, targets)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,  # Important for consistent batch sizes in Hessian computation
    )


def analyze_pattern_completion_accuracy(model, sequences, pattern_info, device):
    """
    Analyze how well the model completes patterns (indicative of induction head formation).

    Returns:
        pattern_accuracy: Accuracy on tokens that are part of repeated patterns
        random_accuracy: Accuracy on random tokens
        per_position_accuracy: Accuracy for each position in repeated patterns
    """
    model.eval()

    pattern_correct = 0
    pattern_total = 0
    random_correct = 0
    random_total = 0

    # Track accuracy by position within pattern
    position_stats = {}

    with torch.no_grad():
        for idx, (seq, patterns) in enumerate(zip(sequences, pattern_info)):
            seq = seq.unsqueeze(0).to(device)

            # Get model predictions
            output = model(seq[:, :-1])  # Don't include last token as input
            predictions = output.argmax(dim=-1).squeeze(0)

            # Mark which positions are part of patterns
            is_pattern = torch.zeros(len(seq[0]) - 1, dtype=torch.bool)

            for pattern_data in patterns:
                positions = pattern_data["positions"]
                length = pattern_data["length"]

                # Only check second and later occurrences
                for pos_idx, pos in enumerate(positions[1:], 1):
                    for j in range(length):
                        if pos + j < len(is_pattern):
                            is_pattern[pos + j] = True

                            # Track position-specific accuracy
                            if j not in position_stats:
                                position_stats[j] = {"correct": 0, "total": 0}

                            position_stats[j]["total"] += 1
                            if predictions[pos + j].item() == seq[0, pos + j + 1].item():
                                position_stats[j]["correct"] += 1

            # Calculate accuracies
            pattern_mask = is_pattern
            random_mask = ~is_pattern

            if pattern_mask.any():
                pattern_correct += (predictions[pattern_mask] == seq[0, 1:][pattern_mask]).sum().item()
                pattern_total += pattern_mask.sum().item()

            if random_mask.any():
                random_correct += (predictions[random_mask] == seq[0, 1:][random_mask]).sum().item()
                random_total += random_mask.sum().item()

    pattern_accuracy = pattern_correct / pattern_total if pattern_total > 0 else 0
    random_accuracy = random_correct / random_total if random_total > 0 else 0

    per_position_accuracy = {}
    for pos, stats in position_stats.items():
        if stats["total"] > 0:
            per_position_accuracy[pos] = stats["correct"] / stats["total"]

    return pattern_accuracy, random_accuracy, per_position_accuracy
