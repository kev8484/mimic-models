import argparse
from pathlib import Path

import torch
from transformers import BertModel, BertConfig, BertTokenizer


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
    parser.add_argument("--bert-dir", type=Path,
                        help="The directory containing pre-trained BERT models", required=True)
    parser.add_argument("--text-data", type=Path,
                        help="File containing text data to embed", required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="The directory to save the word embedding tensors", required=True)
    parser.add_argument("-b", "--batch-size", type=int, default=128,
                        help="Batch size to feed into tokenizer and word embedder")
    parser.add_argument("--seq-length", type=int, default=64,
                        help="Number of tokens in output sequence (will pad or truncate as needed)")
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args(args)

    # load BERT
    print("Loading pre-trained BERT...")
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    bert_config = BertConfig.from_json_file(
        args.bert_dir.glob("*config.json")[0])
    model = BertModel.from_pretrained(
        args.bert_dir.glob("*model.bin")[0], config=bert_config)
    # tokenize text
    tokens = tokenize(
        tokenizer=bert_tokenizer,
        text_data=args.text_data,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )
    print("Tokenization completed.")
    # generate embeddings
    embeddings = embed(model=model, token_batches=tokens)
    print("Embedding completed.")
    # save to disk
    torch.save(embeddings, args.output_dir /
               Path(f"mimic_discharge_summaries_bert_{args.seq_length}tkns.pt"))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])