import argparse
from pathlib import Path

import boto3
import pandas as pd
import torch
from transformers import BertModel, BertConfig, BertTokenizer


S3_BUCKET = "mimic-deeplearning-text-cnn"
S3_BERT_CONFIG = "bert/bert_config.json"
S3_BERT_MODEL = "bert/pytorch_model.bin"
S3_BERT_VOCAB = "bert/vocab.txt"
S3_DATA_DIR = "s3://mimic-deeplearning-text-cnn/data/"


def fetch_bert(s3_bucket):
    # create local directory to store fetched objects
    local_bert_dir = Path.cwd() / Path("bert")
    local_bert_dir.mkdir(exist_ok=True)
    # define target file names (s3 client requires string formatted paths)
    local_config = str(local_bert_dir) + '/bert_config.json'
    local_model = str(local_bert_dir) + '/pytorch_model.bin'
    local_vocab = str(local_bert_dir) + '/vocab.txt'
    # download objects
    s3_bucket.download_file(S3_BERT_CONFIG, local_config)
    s3_bucket.download_file(S3_BERT_MODEL, local_model)
    s3_bucket.download_file(S3_BERT_VOCAB, local_vocab)

    # Note: tokenizer requires directory containing vocab file, not
    # the vocab file itself
    return str(local_bert_dir), local_config, local_model


def load_bert(bert_dir, bert_config, bert_model):
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    config = BertConfig.from_pretrained(bert_config)
    model = BertModel.from_pretrained(bert_model, config=config)

    return tokenizer, model


def tokenize(tokenizer, text_file, batch_size=128, seq_length=64):
    token_li = []
    text_data = pd.read_csv(text_file)['TEXT']
    for i in range(text_data.shape[0] // batch_size + 1):
        start_idx = i * batch_size
        stop_idx = (i + 1) * batch_size

        if i % 100 == 0:
            print(f"Tokenizing batch {i + 1}...")
        tokens = tokenizer(
            text_data[start_idx:stop_idx].values.tolist(),
            padding='max_length',
            truncation=True,
            max_length=seq_length,
            return_tensors="pt",
        )
        token_li.append(tokens)

    return token_li


def embed(model, token_batches):
    model.eval()
    hidden_states = []
    with torch.no_grad():
        for i, batch in enumerate(token_batches):
            if i % 100 == 0:
                print(f"Embedding batch {i + 1}...")
            last_hidden_state, _ = model(**batch)
            hidden_states.append(last_hidden_state)

    embeddings = torch.cat(hidden_states)
    return embeddings


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-dir", type=Path, required=True,
                        help="The directory containing pre-trained BERT models")
    parser.add_argument("--text-data", type=Path,
                        help="File containing text data to embed", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(),
                        help="The directory to save the word embedding tensors")
    parser.add_argument("-b", "--batch-size", type=int, default=128,
                        help="Batch size to feed into tokenizer and word embedder")
    parser.add_argument("--seq-length", type=int, default=128,
                        help="Number of tokens in output sequence (will pad or truncate as needed)")
    parser.add_argument("--aws", action="store_true",
                        help="Store output to AWS S3 bucket")
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args(args)

    # load BERT
    print("Loading pre-trained BERT...")
    if args.aws:
        # fetch objects from s3 bucket
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(S3_BUCKET)
        # returns local paths to downloaded objects
        bert_dir, bert_config, bert_model = fetch_bert(bucket)
    else:
        # retrieve from local directory
        bert_dir = str(args.bert_dir)
        bert_config = list(args.bert_dir.glob("*config.json"))[0]
        bert_model = list(args.bert_dir.glob("*model.bin"))[0]

    # load the tokenizer and model
    bert_tokenizer, model = load_bert(bert_dir, bert_config, bert_model)

    # tokenize text
    print("Tokenizing...")
    tokens = tokenize(
        tokenizer=bert_tokenizer,
        text_file=args.text_data,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )
    # generate embeddings
    print("Embedding...")
    embeddings = embed(model=model, token_batches=tokens)
    # save to disk
    print("Saving...")
    torch.save(embeddings, args.output_dir /
               Path(f"mimic_discharge_summaries_bert_{args.seq_length}tkns.pt"))
    print("Completed.")


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
