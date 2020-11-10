import argparse
from pathlib import Path

import boto3
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


S3_BUCKET = "mimic-deeplearning-text-cnn"
S3_RAW_TEXT = "data/mimic_discharge_summaries_2500chars.csv"


def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    return tokenizer, model


def fetch_raw_data(s3_bucket):
    # create local directory to store fetched objects
    local_data_dir = Path.cwd() / Path("data")
    local_data_dir.mkdir(exist_ok=True)
    # define target file names (s3 API requires string formatted paths)
    local_raw_data = str(local_data_dir) + '/raw_text_data.csv'
    # download object
    s3_bucket.download_file(S3_RAW_TEXT, local_raw_data)

    return local_raw_data


def tokenize(tokenizer, text_file, batch_size=128, seq_length=128):
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


def move_to_GPU(input_ids, token_type_ids, attention_mask):

    input_ids = input_ids.cuda()
    token_type_ids = token_type_ids.cuda()
    attention_mask = attention_mask.cuda()

    output = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
    }

    return output


def embed(model, token_batches):
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    hidden_states = []
    with torch.no_grad():
        for i, batch in enumerate(token_batches):
            if i % 50 == 0:
                print(f"Embedding batch {i + 1}...")
            if torch.cuda.is_available():
                batch = move_to_GPU(**batch)
            last_hidden_state, _ = model(**batch)
            hidden_states.append(last_hidden_state)

    embeddings = torch.cat(hidden_states)
    return embeddings


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-data", type=Path,
                        help="File containing text data to embed")
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

    print("Loading pre-trained BERT...")

    # load the BERT tokenizer and model
    bert_tokenizer, model = load_bert()
    # fetch raw text csv from S3 or get from local path
    if args.text_data is None:
        # fetch objects from s3 bucket
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(S3_BUCKET)
        raw_data = fetch_raw_data(bucket)
    else:
        raw_data = args.text_data
    # tokenize text
    print("Tokenizing...")
    tokens = tokenize(
        tokenizer=bert_tokenizer,
        text_file=raw_data,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )
    # generate embeddings
    print("Embedding...")
    embeddings = embed(model=model, token_batches=tokens)
    # save to disk
    print("Saving locally...")
    outfile = str(args.output_dir) + \
        f"/mimic_discharge_summaries_bert_{args.seq_length}tkns.pt"
    torch.save(embeddings, outfile)

    if args.aws:
        print("Uploading to S3 bucket...")
        s3.upload_file(
            outfile, S3_BUCKET, f"data/mimic_discharge_summaries_pubmed_bert_{args.seq_length}tkns.pt")
    print("Completed.")


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
