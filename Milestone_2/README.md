# Question & Answering Task 🚀

## 📜 Overview

This project implements and compares three neural QA models on the SQuAD dataset, aiming to predict exact answer spans within a given context.

## 🛠️ Requirements

* Python 3.8+
* PyTorch
* NumPy, pandas
* scikit-learn
* GloVe embeddings

---

## 🤖 Approach 1: Transformer-Based QA Model

* **Architecture:** Transformer encoder + cross-attention
* **Embedding:** Learned embeddings + positional encodings
* **Training:** 30 epochs, batch size 16, Adam lr=0.001, dropout=0.1
* **Loss:** Weighted cross-entropy on start/end indices

**Results (Validation):**

| Epochs | Text F1 | EVM   | Start F1 | End F1 |
| ------ | ------- | ----- | -------- | ------ |
| 5      | 15.97%  | 5.58% | 4.62%    | 12.52% |
| 30     | 13.43%  | 3.81% | 3.55%    | 8.25%  |

---

## 🤖 Approach 2: Simplified Transformer QA

* **Simplifications:** Removed pre-encoding, added ReLU in FFN
* **Embedding:** GloVe + trainable unk embeddings
* **Training:** 30 epochs, batch size 16, same optimizer
* **Enhancement:** Length-aware loss to penalize overly long answers

**Results (Validation):**

| Epochs | Text F1 | EVM   | Start F1 | End F1 |
| ------ | ------- | ----- | -------- | ------ |
| 10     | 21.88%  | 7.48% | 8.71%    | 13.88% |
| 30     | 17.38%  | 5.03% | 6.61%    | 11.21% |

> 🔍 *Note: Best text-F1 at 10 epochs; performance drops with overfitting beyond.*

---

## 🤖 Approach 3: Co-Attention BiLSTM QA

* **Embedding:** GloVe + trainable unk embeddings
* **Encoder:** Co-attention affinity + Gaussian smoothing
* **Decoder:** BiLSTM predicting start/end spans
* **Training:** 15 epochs, batch size 32, Adam lr=0.001, dropout=0.3

**Results (Validation):**

| Epochs | Text F1 | EVM   |
| ------ | ------- | ----- |
| 5      | 15.62%  | 6.80% |
| 15     | 16.12%  | 7.76% |

> ⚠️ *Limitation: Data size and shallow BiLSTM depth constrained performance.*

---

## 📈 Visualizations

* Answer length vs. distribution & F1
* Question-type performance (what, who, when, ...)

*Refer to the report for detailed figures.*

---

## 🚀 Usage

1. Prepare SQuAD dataset in `data/` directory.
2. Run training script:

   ```bash
   python train.py --model QATransformerBasedModel
   python train.py --model CoAttentionWithBiLSTMModel
   ```

---

*Prepared by Mazen Soliman & Mohamed Shamekh* 😊
