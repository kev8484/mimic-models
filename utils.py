import random
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (TensorDataset, Subset, DataLoader,
                              WeightedRandomSampler, RandomSampler, SequentialSampler)


def set_seed(seed_value=42):
    """ Set seed for reproducibility.
    """

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def load_tokens(input_id_path, token_type_id_path, attention_mask_path, label_path):
    """ Load BERT tokens (comprised of 3 diff tensors)
    """
    input_ids = torch.load(input_id_path)
    token_type_ids = torch.load(token_type_id_path)
    attention_mask = torch.load(attention_mask_path)
    labels = torch.load(label_path)

    return TensorDataset(input_ids, token_type_ids, attention_mask, labels), labels


def load_embeddings(data_path, label_path):
    """ Load word embedding vector
    """
    data = torch.load(data_path)
    labels = torch.load(label_path)

    return TensorDataset(data, labels), labels


def train_val_test_split(dataset, train_size=0.8, random_seed=42):
    """ Split dataset into training, validation, and test sets
    """

    full_len = len(dataset)
    train_len = int(np.floor(train_size * full_len))
    val_len = (full_len - train_len) // 2
    test_len = full_len - train_len - val_len

    shuffled_indices = np.random.permutation(full_len)
    train_indices = shuffled_indices[:train_len]
    val_indices = shuffled_indices[train_len:(val_len + train_len)]
    test_indices = shuffle_indices[(val_len + train_len):]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    assert len(train_dataset) == train_len and len(
        val_dataset) == val_len and len(test_dataset) == test_len

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    dataset,
    labels,
    batch_size=32,
    train_size=0.8,
    random_seed=42,
    num_workers=2,
    balance=False,
):
    """ Return data loaders for train/val/test sets
    """

    train_dataset, val_dataset, test_dataset = train_val_test_split(
        dataset,
        train_size=train_size,
        random_seed=random_seed,
    )
    # get randomized batches for training, optionally weight to balance the classes
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=balanced_sampler(
            labels, random_seed=random_seed) if balance else RandomSampler(train_dataset),
        num_workers=num_workers,
    )
    # no need to for random sampling or class balancing when evaluating the validation set
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(val_dataset),
        num_workers=num_workers,
    )
    # no need to for random sampling or class balancing when evaluating the test set
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(test_dataset),
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def balanced_sampler(labels, random_seed=None):
    """ Return a sampler to balance class weights
    """
    _, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    sample_weights = weights[labels]

    if random_seed is not None:
        generator = torch.Generator().manual_seed(random_seed)
    else:
        generator = None

    sampler = WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return sampler


def save_model_state(save_dir, step, states):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = f"best_steps_{step}.pt"
    full_path = Path(save_dir) / Path(filename)
    torch.save(states, str(full_path))


def load_model_state(load_dir):
    files = sorted(
        Path(load_dir).glob("best_steps_*.pt"),
        key=lambda f: f.stat().st_mtime
    )
    latest = files[-1]
    logging.info(f"Loading the model state from: {latest}")
    return torch.load(latest)
