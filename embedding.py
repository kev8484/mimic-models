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


def flatten_token_batches(tokens_li):
    """ Convert list of dicts to tensors
    """

    input_ids = torch.cat([t.input_ids for t in tokens_li])
    token_type_ids = torch.cat([t.token_type_ids for t in tokens_li])
    attention_mask = torch.cat([t.attention_mask for t in tokens_li])

    return input_ids, token_type_ids, attention_mask


def format_filepath(output_dir, seq_len, file_type):
    """ Helper function for formatting output filenames.
    """
    return f"{output_dir}/mimic_discharge_summaries_bert_{seq_len}tkns_{file_type}.pt"


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
                        help="File containing text data to tokenize/embed")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(),
                        help="The directory to save the output tensors")
    parser.add_argument("-embeddings", action="store_true",
                        help="Return word embedding tensors. Otherwise, return tokens")
    parser.add_argument("-b", "--batch-size", type=int, default=128,
                        help="Batch size")
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
        raise RuntimeError("Must use --text-data to include a csv file with discharge notes")
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

    output_files = []

    if not args.embeddings:
        # save token tensors locally
        input_ids, token_type_ids, attention_mask = flatten_token_batches(
            tokens)
        print("Saving locally...")
        for tensor, file_type in zip([input_ids, token_type_ids, attention_mask],
                                     ['input_ids', 'token_type_ids', 'attention_mask']):
            outpath = format_filepath(
                str(args.output_dir), args.seq_length, file_type)
            torch.save(tensor, outpath)
            output_files.append(outpath)

    else:
        # generate embeddings
        print("Embedding...")
        embeddings = embed(model=model, token_batches=tokens)
        # save to disk
        print("Saving locally...")
        outpath = format_filepath(
            str(args.output_dir), args.seq_length, "embeddings")
        torch.save(embeddings, outpath)
        output_files.append(outpath)

    if args.aws:
        print("Uploading to S3 bucket...")
        s3 = boto3.resource('s3')

        for local_path in output_files:
            # get just the file name from the path
            filename = Path(local_path).name
            # save to s3 bucket in directory called 'data'
            s3.upload_file(local_path, S3_BUCKET, f"data/{filename}")

    print("Completed.")


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
