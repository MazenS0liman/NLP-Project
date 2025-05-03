#!/bin/bash

mkdir -p data
mkdir -p glove

if which -s wget; then
  WGET_CMD='wget'
  SAVE_ARG=''
  SAVE_TO_ARG='-O'
else
  WGET_CMD='curl -L' # -L to follow redirects
  SAVE_ARG='-O'
  SAVE_TO_ARG='-o'
fi

# Download SQuAD dataset
$WGET_CMD https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz $SAVE_TO_ARG data/squad_train.jsonl.gz
$WGET_CMD https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz $SAVE_TO_ARG data/squad_dev.jsonl.gz

# Download GloVe vectors
$WGET_CMD http://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.6B.zip $SAVE_ARG
unzip glove.6B.zip
mv glove.6B.*.txt glove/
rm glove.6B.zip

echo
echo "done!"
