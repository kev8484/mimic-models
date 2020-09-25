import argparse
from pathlib import Path

import torch


def tokenize(tokenizer, text_data, batch_size=128, seq_length=64):
    token_li = []
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
    parser.add_argument("--mimic-dir", type=Path,
                        help="The directory containing MIMIC data files.", required=True)
    parser.add_argument("-b", "--batch-size", type=int, default=128,
                        help="Batch size to feed into tokenizer and word embedder")
    parser.add_argument("--seq-length", type=int, default=64,
                        help="Number of tokens in output sequence (will pad or truncate as needed)")
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args(args)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
