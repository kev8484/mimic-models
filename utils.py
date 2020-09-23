import torch


def load_dataset(data_path, labels_path, batch_size):
    data = torch.load(data_path)
    labels = torch.load(labels_path)

    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, labels, train_size=0.8, random_seed=42)

    train_data, val_data, train_labels, val_labels = \
        train_test_split(train_data, train_labels,
                         train_size=0.8, random_seed=42)

    train_loader = tensors_to_loader(
        train_data, train_labels, batch_size=batch_size)
    val_loader = tensors_to_loader(val_data, val_labels, batch_size=batch_size)
    test_loader = tensors_to_loader(
        test_data, test_labels, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def tensors_to_loader(data, labels, batch_size=32):
    # convert tensors to pytorch dataset
    dataset = torch.utils.data.TensorDataset(
        data.float(),
        labels.long(),
    )

    # Create iterable
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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
        (train_size, test_size),
        generator=generator
    )

    train_labels, test_labels = torch.utils.data.random_split(
        labels,
        (train_size, test_size),
        generator=generator
    )

    return train_data, test_data, train_labels, test_labels
