#!/usr/bin/env python
"""
Train a Question Answering model using PyTorch, supporting both CoAttention+BiLSTM and Transformer models.
"""
import os
import copy
import nltk
import torch
import argparse
import numpy as np
import pandas as pd
from model import CoAttentionWithBiLSTMModel, QATransformerBasedModel
import torch.nn.functional as F
from data import QADataset, Vocabulary, Tokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from evaluate import compute_em, compute_f1
from utils import enforce_position_constraints

nltk.download('wordnet')
nltk.download('omw-1.4')

def parse_args():
    """
    Parse command line arguments for training a QA model.
    """

    parser = argparse.ArgumentParser(description='Train a QA model')

    parser.add_argument('--model_name', type=str, default='CoAttentionWithBiLSTMModel',
                        choices=['CoAttentionWithBiLSTMModel', 'QATransformerBasedModel'],
                        help='Which QA architecture to train')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Epoch count')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial LR')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding size')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for LSTM')
    parser.add_argument('--num_heads', type=int, default=8, help='Transformer heads')
    parser.add_argument('--d_ff', type=int, default=512, help='Transformer feed-forward dim')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_context_len', type=int, default=400, help='Max context length')
    parser.add_argument('--checkpoint_dir', type=str, default='./model/', help='Checkpoint dir')
    parser.add_argument('--out_dir', type=str, default='./out/', help='Output dir')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare datasets and vocab
    train_ds = QADataset(path=args.train_data)
    val_ds = QADataset(path=args.val_data)
    vocab = Vocabulary(train_ds.samples, vocab_size=200000)
    tokenizer = Tokenizer(vocab)
    train_ds.tokenizer = tokenizer
    val_ds.tokenizer = tokenizer

    # Filter by length
    train_ds.samples = train_ds.samples[train_ds.samples['context'].map(len) <= args.max_context_len].reset_index(drop=True)
    val_ds.samples = val_ds.samples[val_ds.samples['context'].map(len) <= args.max_context_len].reset_index(drop=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds._collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=val_ds._collate_batch)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Metrics storage
    train_losses = []
    val_em_scores = []
    val_f1_scores = []

    # Instantiate model
    if args.model_name == 'CoAttentionWithBiLSTMModel':
        model = CoAttentionWithBiLSTMModel(
            vocab=vocab,
            vocab_decoder=vocab.decoding,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=args.hidden_size * 2,
            dropout_prob=args.dropout
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    else:  # Transformer
        model = QATransformerBasedModel(
            vocab=vocab,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            dropout_prob=args.dropout
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Training & Validation
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            if batch is None:
                continue
            contexts = batch['context'].to(device)
            questions = batch['question'].to(device)
            start_pos = batch['answer_start'].to(device)
            end_pos = batch['answer_end'].to(device)

            optimizer.zero_grad()
            start_logits, end_logits = model(contexts, questions)
            end_logits = enforce_position_constraints(end_logits, start_pos)

            loss_start = F.cross_entropy(start_logits, start_pos)
            loss_end = F.cross_entropy(end_logits, end_pos)
            loss = 0.5 * loss_start + 0.5 * loss_end
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch}/{args.num_epochs} — Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        em_scores, f1_scores = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                contexts = batch['context'].to(device)
                questions = batch['question'].to(device)
                start_pos = batch['answer_start'].to(device)
                end_pos = batch['answer_end'].to(device)

                start_logits, end_logits = model(contexts, questions)
                end_logits = enforce_position_constraints(end_logits, start_pos)
                start_pred = start_logits.argmax(dim=1)
                end_pred = end_logits.argmax(dim=1)

                # Decode per sample
                for i in range(contexts.size(0)):
                    tok_ids = contexts[i].cpu().tolist()
                    tokens = tokenizer.convert_ids_to_tokens(tok_ids)
                    s, e = start_pred[i].item(), end_pred[i].item()
                    if e < s: e = s
                    pred = " ".join(tokens[s:e+1]).strip()
                    gold = val_ds.samples.iloc[i]['answers']['text'][0].strip()
                    em_scores.append(compute_em(pred, gold))
                    f1_scores.append(compute_f1(pred, gold))

        mean_em = np.mean(em_scores) * 100
        mean_f1 = np.mean(f1_scores) * 100
        val_em_scores.append(mean_em)
        val_f1_scores.append(mean_f1)
        print(f"Epoch {epoch} — Val EM: {mean_em:.2f}%, Val F1: {mean_f1:.2f}%")

        # Checkpoint every 5
        if epoch % 5 == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_epoch{epoch}.pt")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'vocab': vocab}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save metrics
    metrics = pd.DataFrame({
        'epoch': list(range(1, args.num_epochs+1)),
        'train_loss': train_losses,
        'val_em': val_em_scores,
        'val_f1': val_f1_scores
    })
    metrics_path = os.path.join(args.out_dir, 'training_metrics.csv')
    metrics.to_csv(metrics_path, index=False)
    print(f"Stored metrics at {metrics_path}")
