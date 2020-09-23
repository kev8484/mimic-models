import torch


def load_dataset(data_path, labels_path, batch_size):
    data = torch.load(data_path)
    labels = torch.load(labels_path)

    tensors = torch.utils.data.TensorDataset(
        data.float(),
        labels.long(),
    )

    # Create iterable
    loader = torch.utils.data.DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    return loader
