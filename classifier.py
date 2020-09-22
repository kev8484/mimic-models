import argparse

import torch
import torch.nn as nn

from model import TextCNN
import config


def train(args):
    model = TextCNN(
        num_classes=2,
        embedding_size=768,
        num_filters=128,
        dropout_rate=0.5,
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0
        running_corrects = 0
        for i, data in enumerate(args.trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            probs, classes = model(inputs)
            # backprop
            loss = loss_function(probs, labels)
            loss.backward()
            # update/optimize
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_corrects += torch.sum(classes == labels.data)
            if i % 50 == 0 and i != 0:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f acc %.3f' %
                      (epoch + 1, i + 1, running_loss / 50, running_corrects / 50))
                running_loss = 0.0
                running_corrects = 0.0

    print('Finished Training')


def parse_args(args):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    def add_common_arguments(parser):
        parser.add_argument("-m", "--model_dir", type=str, required=True,
                            help="The directory where a trained model is saved.")
        parser.add_argument("-c", "--config_file",
                            help="The path to the configuration file.")

    # For train mode
    parser_train = subparsers.add_parser("train", help="train a model")
    add_common_arguments(parser_train)
    parser_train.add_argument("--checkpoint_interval", default=1000, type=int,
                              help="The period at which a checkpoint file will be created.")
    parser_train.add_argument("--keep_checkpoint_max", default=5,
                              type=int, help="The number of checkpoint files to be preserved.")
    parser_train.add_argument("--summary_interval", default=100,
                              type=int, help="The period at which summary will be saved.")

    # For predict mode
    parser_predict = subparsers.add_parser(
        "predict", help="predict by using a trained model")
    add_common_arguments(parser_predict)
    # parser_predict.add_argument("file", type=argparse.FileType("r"), help="An input text file.")

    args = parser.parse_args()

    return args


def main(args):
    args = parse_args(args)

    if args.subcommand == "train":
        config.load(args.config_file)
        config.save(args.model_dir)
        states = None
        train(args, states)

    elif args.subcommand == "predict":
        pass


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
