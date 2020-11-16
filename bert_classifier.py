import argparse
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from transformers import BertForSequenceClassification

from utils import load_dataset, save_model_state
from toml_config import Config


def train(args, states=None):

    config_obj = Config(args.config_file)
    config = config_obj.elements
    logging.info("Loading datasets...")
    train_loader, val_loader, test_loader = load_dataset(
        data_path=config['data'],
        labels_path=config['labels'],
        batch_size=config['batch_size'],
        random_seed=config['random_seed'],
        balance=config['correct_imbalance'],
    )

    model = BertForSequenceClassification.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )

    if torch.cuda.is_available():
        model.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    best_metric = 0

    # loop over the dataset multiple times
    for epoch in range(1, config['num_epochs'] + 1):
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
            _, logits = model(inputs)
            probs = F.softmax(logits, dim=1)

            # backprop
            loss = loss_function(probs, labels)
            loss.backward()
            # update/optimize
            optimizer.step()

            # Log summary
            running_losses.append(loss.item())
            if i % args.log_interval == 0:
                interval_loss = sum(running_losses) / len(running_losses)
                logging.info(f"step = {i}, loss = {interval_loss}")
                running_losses = []

            if i % args.test_interval == 0:
                dev_metric = eval(
                    val_loader,
                    model,
                    loss_function,
                    args.eval_metric,
                )
                if dev_metric > best_metric:
                    best_metric = dev_metric
                    states = {
                        "epoch": epoch,
                        "step": i,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
                    save_model_state(
                        save_dir=args.model_dir,
                        step=i,
                        states=states
                    )

    print(f"Finished Training, best {args.eval_metric}: {best_metric}")


def eval(data_iter, model, loss_function, metric):
    model.eval()

    corrects, avg_loss = 0, 0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in data_iter:
            inputs, labels = batch

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            _, logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            classes = torch.max(probs, 1)[1]  # (batch_size)

            loss = loss_function(probs, labels)
            # accuracy
            avg_loss += loss.item()
            corrects += torch.sum(classes == labels.data)
            # auc
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())

        size = len(data_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * corrects / size

        y_true = np.concatenate(all_labels)
        y_score = np.concatenate(all_probs)
        auc = roc_auc_score(y_true, y_score)

    logging.info(
        f"\nEvaluation - loss: {avg_loss:.6f}  acc: {accuracy:.4f}%({corrects}/{size})  auc: {auc}\n")

    if metric == "auc":
        return auc

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
    parser_train.add_argument("--log-interval", default=30,
                              type=int, help="Number of batches to print summary")
    parser_train.add_argument("--test-interval", default=50,
                              type=int, help="Number of batches to run validation test")
    parser_train.add_argument("--eval-metric", default="accuracy",
                              type=str, help="Metric used for determing the best model")

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
        # toml_config.load(args.config_file)
        # toml_config.save(args.model_dir)
        states = None
        train(args, states)

    elif args.subcommand == "predict":
        pass


if __name__ == '__main__':
    logging.basicConfig(
        format="%(levelname)s\t%(asctime)s\t%(message)s",
        level=logging.INFO
    )
    import sys
    main(sys.argv[1:])