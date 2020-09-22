import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """ Text CNN from Yoon Kim's 2014 paper: https://arxiv.org/pdf/1408.5882.pdf

    Params
    ------
    embedding_size: int
        The vector length for word embeddings. (word2vec is 300, BERT is 768)
    num_filters: int
        The number of filters per kernel to apply. (Default = 128)
    num_classes : int
        Number of output classes. (Default = 2)
    kernel_sizes: tuple(int)
        The kernel size for each desired filter(Default = (3,4,5))
    dropout_rate: float
        Probability of dropping a neuron in the dropout layer.  Must be in the range [0.0, 1.0]
        (Default = 0.5)
    embedding_weights: torch.FloatTensor
        Pre-trained embedding weights to optionally pass.
        (Default is None)
    vocab_size : int
        The number of unique tokens in vocabulary.

    Returns
    -------
    model : nn.Module
    """

    def __init__(
        self,
        embedding_size,
        num_filters=128,
        num_classes=2,
        kernel_sizes=(3, 4, 5),
        dropout_rate=0.5,
        embedding_weights=None,
        vocab_size=None
    ):
        super(TextCNN, self).__init__()

        # word embedding
        # learned (i.e. input tensors are tokens, model will learn embeddings)
        if embedding_weights is None and vocab_size is not None:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embedding_size)
        # pre-trained (i.e. input tensors are tokens, model won't modify embeddings)
        elif embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weights)
        # none (i.e. input tensors are already embedded vectors)
        else:
            self.embedding = None

        # convolutional layer
        self.convs = nn.ModuleList([nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(kernel_size, embedding_size),
        ) for kernel_size in kernel_sizes])

        # dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # fully connected layer
        self.fc = nn.Linear(
            in_features=num_filters * len(kernel_sizes),
            out_features=num_classes
        )

    def forward(self, x):
        if self.embedding is not None:
            # (batch_size, sequence_length, embedding_size)
            x = self.embedding(x)

        # (batch_size, in_channels, sequence_length, embedding_size)
        x = x.unsqueeze(1)

        x_li = []
        for conv in self.convs:
            # (batch_size, out_channels, sequence_length, 1)
            _x = F.relu(conv(x))
            _x = _x.squeeze(3)  # (batch_size, out_channels, sequence_length)
            _x = F.max_pool1d(_x, _x.size(2)).squeeze(
                2)  # (batch_size, out_channels)
            x_li.append(_x)

        x = torch.cat(x_li, 1)

        x = self.dropout(x)  # (batch_size, len(kernel_sizes) * out_channels)
        logits = self.fc(x)  # (batch_size, num_classes)

        probs = F.softmax(logits)  # (batch_size, num_classes)
        classes = torch.max(probs, 1)[1]  # (batch_size)

        return probs, classes
