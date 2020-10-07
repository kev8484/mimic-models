import argparse
import logging

import torch
import torch.nn as nn

from model import TextCNN
from utils import load_dataset, save_model_state
import config


def train(args, states=None):

    train_loader, val_loader, test_loader = load_dataset(
        data_path=config.files.data,
        labels_path=config.files.labels,
        batch_size=config.train.batch_size,
        random_seed=config.train.random_seed,
        balance=config.train.correct_imbalance,
    )

    model = TextCNN(
        num_classes=config.train.num_classes,
        embedding_size=config.train.embedding_size,
        num_filters=config.train.num_filters,
        dropout_rate=config.train.dropout,
    )
    if torch.cuda.is_available():
        model.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    best_acc = 0

    # loop over the dataset multiple times
    for epoch in range(1, config.train.num_epochs + 1):
        logging.info(
            f"==================== Epoch: {epoch} ====================")
        running_losses = []
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            probs, classes = model(inputs)
            # backprop
            loss = loss_function(probs, labels)
            loss.backward()
            # update/optimize
            optimizer.step()

            # Log summary
            running_losses.append(loss.data[0])
            if i % args.log_interval == 0:
                interval_loss = sum(running_losses) / len(running_losses)
                logging.info(f"step = {i}, loss = {interval_loss}")
                running_losses = []

            if i % args.test_interval == 0:
                dev_acc = eval(val_loader, model, loss_function)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    states = {
                        "epoch": epoch,
                        "step": i,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
                    save_model_state(
                        save_dir=args.model_dir,
                        step=i,
                        state=states
                    )

    print(f"Finished Training, best accuracy: {best_acc}")


def eval(data_iter, model, loss_function):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        inputs, labels = batch

        probs, classes = model(inputs)
        loss = loss_function(probs, labels)

        avg_loss += loss.item()
        corrects += torch.sum(classes == labels.data)

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    logging.info(
        f"\nEvaluation - loss: {avg_loss:.6f}  acc: {accuracy:.4f}%({corrects}/{size}) \n")
    return accuracy


def parse_args(args):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    def add_common_arguments(parser):
        parser.add_argument("-m", "--model-dir", type=str, required=True,
                            help="The directory where a trained model is saved")
        parser.add_argument("-c", "--config-file",
                            help="The path to the configuration file")

    # For train mode
    parser_train = subparsers.add_parser("train", help="train a model")
    add_common_arguments(parser_train)
    parser_train.add_argument("--checkpoint-interval", default=1000, type=int,
                              help="The period at which a checkpoint file will be created")
    parser_train.add_argument("--keep-checkpoint-max", default=5,
                              type=int, help="The number of checkpoint files to be preserved")
    parser_train.add_argument("--log-interval", default=50,
                              type=int, help="Number of batches to print summary")
    parser_train.add_argument("--test-interval", default=100,
                              type=int, help="Number of batches to run validation test")

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
    logging.basicConfig(
        format="%(levelname)s\t%(asctime)s\t%(message)s", level=logging.INFO)
    import sys
    main(sys.argv[1:])
