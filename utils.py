import numpy as np
import torch


def load_dataset(data_path, labels_path, batch_size):
    """ TODO: make a sensible interface for the random_seed used throughout
    """
    data = torch.load(data_path)
    labels = torch.load(labels_path)

    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, labels, train_size=0.8, random_seed=42)

    train_data, val_data, train_labels, val_labels = \
        train_test_split(train_data, train_labels,
                         train_size=0.8, random_seed=42)

    train_loader = tensors_to_loader(
        data=train_data,
        labels=train_labels,
        batch_size=batch_size,
        balance=True
    )
    val_loader = tensors_to_loader(
        data=val_data,
        labels=val_labels,
        batch_size=batch_size,
        balance=True
    )
    test_loader = tensors_to_loader(
        data=test_data,
        labels=test_labels,
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader


def tensors_to_loader(data, labels, batch_size=32, balance=False):
    # convert tensors to pytorch dataset
    dataset = torch.utils.data.TensorDataset(
        data.float(),
        labels.long(),
    )
    # if we're evening out imbalanced classes
    if balance:
        sampler = balanced_sampler(labels, random_seed=42)
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


def train_test_split(data, labels, train_size=0.8, random_seed=42):
    """ Split data into training and test sets.
    """
    train_len = int(train_size * len(data))
    test_len = len(data) - train_size

    if random_seed is not None:
        generator = torch.Generator().manual_seed(random_seed)
    else:
        generator = None

    train_data, test_data = torch.utils.data.random_split(
        data,
        (train_len, test_len),
        generator=generator
    )

    train_labels, test_labels = torch.utils.data.random_split(
        labels,
        (train_len, test_len),
        generator=generator
    )

    return train_data, test_data, train_labels, test_labels


def balanced_sampler(labels, random_seed=42):
    """ Return a sampler to balance class weights
    """
    _, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    sample_weights = weights[labels]

    if random_seed is not None:
        generator = torch.Generator().manual_seed(random_seed)
    else:
        generator = None

    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return sampler
