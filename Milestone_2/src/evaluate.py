#!/usr/bin/env python
'''
This script evaluates the performance of a model by comparing its predictions against ground truth values.
It computes the Exact Match (EM) and F1 scores for the predictions, which are common metrics used in natural language processing tasks, especially in question answering systems.
'''

def compute_em(predicted, actual):
    return int(predicted.strip().lower() == actual.strip().lower())

def compute_f1(predicted, actual):
    pred_tokens = predicted.strip().lower().split()
    actual_tokens = actual.strip().lower().split()

    common = set(pred_tokens) & set(actual_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(actual_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1
