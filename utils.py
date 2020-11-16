import random
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (TensorDataset, random_split, DataLoader,
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

    train_dataset, val_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(random_seed),
    )

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


def load_dataset(data_path, labels_path, batch_size, random_seed=None, balance=False, data_type='embeddings'):
    """ Return data loaders for train/val/test sets
    """
    data = torch.load(data_path)
    labels = torch.load(labels_path)

    train_loader = tensors_to_loader(
        data=train_data,
        labels=train_labels,
        batch_size=batch_size,
        random_seed=random_seed,
        balance=balance
    )
    val_loader = tensors_to_loader(
        data=val_data,
        labels=val_labels,
        batch_size=batch_size,
        random_seed=random_seed,
        balance=balance
    )
    # note: we never apply class weighting to the test set
    test_loader = tensors_to_loader(
        data=test_data,
        labels=test_labels,
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader


def tensors_to_loader(data, labels, batch_size=32, random_seed=None, balance=False):
    # convert tensors to pytorch dataset
    dataset = torch.utils.data.TensorDataset(
        data.float(),
        labels.long(),
    )
    # if we're evening out imbalanced classes
    if balance:
        sampler = balanced_sampler(labels, random_seed=random_seed)
    else:
        sampler = None
    # Create iterable
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2
    )

    return loader


# def train_test_split(data, labels, train_size=0.8, random_seed=None):
#     """ Split data into training and test sets.
#     """
#     train_len = int(np.floor(train_size * data.shape[0]))
#     test_len = data.shape[0] - train_len

#     if data.shape[0] != labels.shape[0]:
#         raise ValueError(
#             f"Input data length ({data.shape[0]}) and input label length ({labels.shape[0]}) don't match")

#     if random_seed is not None:
#         np.random.seed(random_seed)

#     # randomly assign indices to training and test sets
#     shuffle_indices = np.random.permutation(data.shape[0])
#     train_indices, test_indices = shuffle_indices[:
#                                                   train_len], shuffle_indices[train_len:]

#     if len(train_indices) != train_len or len(test_indices) != test_len:
#         raise ValueError(f"Random sampling produced inconsistent data sets.")

#     train_data = data[train_indices, :]
#     train_labels = labels[train_indices]

#     test_data = data[test_indices, :]
#     test_labels = labels[test_indices]

#     return train_data, test_data, train_labels, test_labels


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
