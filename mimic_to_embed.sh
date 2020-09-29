#!/bin/bash

MIMIC_DIR="$HOME/mimic/data"
mkdir -p "$MIMIC_DIR"
# get mimic dataset from s3 bucket onto ec2 instance
aws s3 cp "s3://mimic-deeplearning-text-cnn/mimic/data/*.csv $MIMIC_DIR"

python preprocessing.py --mimic-dir "$MIMIC_DIR"

BERT_DIR="$HOME/bert/model"
mkdir -p "$BERT_DIR"
# get bert models from s3 bucket onto ec2 instance
aws s3 cp "s3://mimic-deeplearning-text-cnn/bert/model/* $BERT_DIR"

