#!/usr/bin/env python
'''
This module provides a Dataset class for a question-answering task,
along with a Vocabulary and Tokenizer class for handling text data.
'''

import re
import nltk
import torch 
import itertools
import collections
import pandas as pd
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset
from utils import load_embeddings, tokenize_with_bert

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

lemmatizer = WordNetLemmatizer()

###############################################################################
#                                                                             #
#                               Vocabulary                                    #
#                                                                             #
###############################################################################
class Vocabulary:
    """
    Creates mappings for words → indices and indices → words.
    """
    def __init__(self, samples, vocab_size):
        self.samples = samples
        self.vocab_size = vocab_size
        self.words = self._initialize(samples, vocab_size)
        self.encoding = {word: idx for idx, word in enumerate(self.words)}
        self.decoding = {idx: word for idx, word in enumerate(self.words)}

    def _initialize(self, samples, vocab_size):
        """Build vocabulary with lemma support"""
        embedding_map = load_embeddings("/kaggle/input/glove/other/default/1/glove.6B.300d.txt")
        vocab_counts = collections.defaultdict(int)
        
        for _, row in samples.iterrows():
            # Get base tokens
            tokens = re.findall(r"\w+(?:[-']\w+)*", row['context'].lower()) + \
                     re.findall(r"\w+(?:[-']\w+)*", row['question'].lower())
            
            # Count both original and lemma forms
            for token in tokens:
                vocab_counts[token] += 1
                lemma = lemmatizer.lemmatize(token)
                if lemma != token:
                    vocab_counts[lemma] += 0.5  # Partial count for lemmas
        
        # Sort by combined frequency
        sorted_words = sorted(vocab_counts.items(), 
                            key=lambda x: (-x[1], x[0]))[:vocab_size-2]
        
        return [PAD_TOKEN, UNK_TOKEN] + [w[0] for w in sorted_words]
        
    def __len__(self):
        return len(self.words)

###############################################################################
#                                                                             #
#                               Tokenizer                                     #
#                                                                             #
###############################################################################
class Tokenizer:
    """
    Converts lists of words to indices and vice versa.
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = vocabulary.encoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        return [self.vocabulary.encoding.get(token.lower(), self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, token_ids):
        return [self.vocabulary.decoding.get(token_id, UNK_TOKEN) for token_id in token_ids]

###############################################################################
#                                                                             #
#                               QADataset                                     #
#                                                                             #
###############################################################################
class QADataset(Dataset):
    """
    Data generator for a QA task; the JSON file should contain character-level answer indices.
    """
    def __init__(self, path):
        # Load JSON-lines file; each line is a JSON object.
        self.samples = pd.read_json(path, lines=True)
        self.tokenizer = None
        # Default pad token id; updated after tokenizer registration.
        self.pad_token_id = 0

    def _collate_batch(self, batch):
        batch = [sample for sample in batch if sample is not None]
        if len(batch) == 0:
            return None  # All samples failed
    
        max_context_len = max(sample['context'].size(0) for sample in batch)
        max_question_len = max(sample['question'].size(0) for sample in batch)
    
        contexts = torch.stack([
            torch.cat([
                sample['context'],
                torch.full((max_context_len - sample['context'].size(0),), self.pad_token_id, dtype=torch.long)
            ]) for sample in batch
        ])
    
        questions = torch.stack([
            torch.cat([
                sample['question'],
                torch.full((max_question_len - sample['question'].size(0),), self.pad_token_id, dtype=torch.long)
            ]) for sample in batch
        ])
    
        answer_starts = torch.stack([sample['answer_start'] for sample in batch])
        answer_ends = torch.stack([sample['answer_end'] for sample in batch])
    
        return {
            'context': contexts,
            'question': questions,
            'answer_start': answer_starts,
            'answer_end': answer_ends
        }

    def register_tokenizer(self, tokenizer):
        """
        Registers a Tokenizer instance and updates pad token id.
        """
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        context_str = sample['context']
        question_str = sample['question']
        answers = sample['answers']
    
        context_tokens, context_spans = tokenize_with_bert(context_str)
        question_tokens, _ = tokenize_with_bert(question_str)
        context_indices = self.tokenizer.convert_tokens_to_ids(context_tokens)
        question_indices = self.tokenizer.convert_tokens_to_ids(question_tokens)
    
        answer_text = answers['text'][0].strip()
        answer_tokens, _ = tokenize_with_bert(answer_text)
    
        context_tokens_lower = [t.lower() for t in context_tokens]
        answer_tokens_lower = [t.lower() for t in answer_tokens]
    
        token_start, token_end = -1, -1
        for i in range(len(context_tokens_lower) - len(answer_tokens_lower) + 1):
            if context_tokens_lower[i:i+len(answer_tokens_lower)] == answer_tokens_lower:
                token_start = i
                token_end = i + len(answer_tokens_lower) - 1
                break
    
        if token_start == -1 or token_end == -1:
            # skip
            return None
    
        return {
            'context': torch.tensor(context_indices, dtype=torch.long),
            'question': torch.tensor(question_indices, dtype=torch.long),
            'answer_start': torch.tensor(token_start, dtype=torch.long),
            'answer_end': torch.tensor(token_end, dtype=torch.long),
        }
