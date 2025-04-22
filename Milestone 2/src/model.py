#!/usr/bin/env python
'''
Description: This module defines the BiLSTM and QAModel classes for a question answering model.
It includes the architecture of the model, including embedding layers, LSTM layers, and a co-attention mechanism.
It also includes the forward pass method for the model, which processes input data and produces start and end logits for answer prediction.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_embeddings, embed_batch, co_attention, create_embedding_matrix

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################################
#                                                                                                      #
#                                           BiLSTM Model                                               #
#                                                                                                      #
########################################################################################################
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob=0.3):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a bidirectional LSTM layer; note batch_first=True keeps tensors as (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            return_sequence: If True, return the entire sequence; if False, return the last hidden state.

        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Run the input sequence through the LSTM layer
        output, _ = self.lstm(x, (h0, c0))
        
        # Apply dropout to the output of the LSTM layer
        output = self.dropout(output)

        # Return the entire sequence
        return output # Shape: (batch_size, seq_len, hidden_size * 2)

########################################################################################################
#                                                                                                      #
#                                    CoAttentionWithBiLSTMModel                                        #
#                                                                                                      #
########################################################################################################
class CoAttentionWithBiLSTMModel(nn.Module):
    def __init__(self, vocab, vocab_decoder, embedding_dim, hidden_size, num_layers, dropout_prob=0.3):
        super(CoAttentionWithBiLSTMModel, self).__init__()

        self.vocab_decoder = vocab_decoder

        # Embedding map using pre-trained GloVe embeddings
        self.embedding_map = load_embeddings("./glove/glove.6B.300d.txt")

        # Initialize Embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings = len(vocab),
            embedding_dim = embedding_dim,
            padding_idx = vocab.encoding[PAD_TOKEN]
        )

        # Load pretrained weights
        embedding_matrix = create_embedding_matrix(vocab, self.embedding_map)
        self.embedding_layer.weight.data.copy_(embedding_matrix)

        # Freeze only GloVe vectors
        for i, word in enumerate(vocab.words):
            if word in self.embedding_map and word != UNK_TOKEN:
                self.embedding_layer.weight.requires_grad_(False)

        # Context Modeling
        self.start_decoder = BiLSTMModel(embedding_dim * 3, hidden_size, num_layers, dropout_prob=dropout_prob)
        self.end_decoder = BiLSTMModel(embedding_dim * 6, hidden_size, num_layers, dropout_prob=dropout_prob)

        # Prediction Layers - two linear layers for start and end index predictions.
        self.start_linear = nn.Linear(3 * embedding_dim + 2 * hidden_size, 1)
        self.end_linear = nn.Linear(3 * embedding_dim + 2 * hidden_size, 1)
    
    def forward(self, context_ids, question_ids):
        """
        Forward pass that includes contextual encoding.

        Args:
            context_ids (Tensor): shape (batch, context_len)
            question_ids (Tensor): shape (batch, question_len)

        Returns:
            start_logits: Tensor of shape (batch, context_len)
            end_logits: Tensor of question word IDs.
            affinity: (batch, context_len, question_len)
        """
        ### Word Embedding
        context_emb_np = embed_batch(embedding_map=self.embedding_map, embedding_layer=self.embedding_layer, batch_token_ids=context_ids, idx2word=self.vocab_decoder, embed_dim=300)
        question_emb_np = embed_batch(embedding_map=self.embedding_map, embedding_layer=self.embedding_layer, batch_token_ids=question_ids, idx2word=self.vocab_decoder, embed_dim=300)

        # Convert numpy arrays to torch tensors (and ensure they are float type).
        context_emb = torch.from_numpy(context_emb_np).float().contiguous().to(device)
        question_emb = torch.from_numpy(question_emb_np).float().contiguous().to(device)

        # --- Encoder: Contextual Embedding via CoAttention ---
        # passage_attention_context, encoder_out = co_attention(context_emb, question_emb, True)
        _, encoder_out = co_attention(
            context_emb, question_emb, conv=True
        )
        # encoder_out shape: (B, L, 3*embedding_dim)

        # --- Decoder for the Start Index Prediction ---
        start_decoded = self.start_decoder(encoder_out)
        # start_decoded shape: (B, L, 2*hidden_size)

        # Concatenate encoder output with the decoded representation
        start_input = torch.cat([encoder_out, start_decoded], dim=-1)  # (B, L, 3*embedding_dim + 2*hidden_size)
        start_logits = self.start_linear(start_input).squeeze(-1)
        start_probs = F.softmax(start_logits, dim=-1).unsqueeze(-1)

        # Compute weighted summary using start_probs
        start_summary = torch.sum(encoder_out * start_probs, dim=1, keepdim=True)
        start_summary_expanded = start_summary.repeat(1, encoder_out.size(1), 1)  # (B, L, 3*embedding_dim)

        # --- Prepare Features for End Index Prediction ---
        combined_for_end = torch.cat([encoder_out, start_summary_expanded], dim=-1)  # (B, L, 6*embedding_dim)
        
        # --- Decoder for the End Index Prediction ---
        end_decoded = self.end_decoder(combined_for_end)
        
        # Concatenate encoder output with the end decoded representation
        # (Note: ensure arguments to torch.cat are provided as a list)
        end_input = torch.cat([encoder_out, end_decoded], dim=-1)  # (B, L, 3*embedding_dim + 2*hidden_size)
        end_logits = self.end_linear(end_input).squeeze(-1)

        return start_logits, end_logits

########################################################################################################
#                                                                                                      #
#                                       Transformer Blocks                                             #
#                                                                                                      #
########################################################################################################
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for the Transformer model.
    This class implements the multi-head attention mechanism as described in the paper "Attention is All You Need".
    
    Args:
        d_model (Tensor): Dimensionality of the input
        num_heads (int): The number of attention heads to split the input into it.

    Returns:
        context: (Tensor): The context vector after applying attention.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        # Ensure that the model dimension is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Calculate the context vector as a weighted sum of values
        context = torch.matmul(attn_weights, value)

        return context

    def split_heads(self, x):
        """
        Split the input tensor into multiple heads for multi-head attention.
        """
        # Reshape the input to have num_heads for multi-head attention.
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape.
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, query, key, value, mask=None):
        # Apply linear transformations and split heads.
        query = self.split_heads(self.W_q(query))  # (batch_size, num_heads, seq_length, d_k)
        key = self.split_heads(self.W_k(key))  # (batch_size, num_heads, seq_length, d_k)
        value = self.split_heads(self.W_v(value))  # (batch_size, num_heads, seq_length, d_k)

        # Perform scaled dot-product attention
        context = self.scaled_dot_product_attention(query, key, value, mask)  # (batch_size, num_heads, seq_length, d_k)

        # Combine the heads and apply output transformation.
        output = self.W_o(self.combine_heads(context))

        return output
    
class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward netowrk consists of two linear transformations with a ReLU activation in between.
    In the context of transformer models, this feed-forward network is applied to each position separately and identically.
    It helps in transforming the features learned by the attention mechanisms within the transformer, 
    acting as an additional processing step for the attention outputs.
    """
    def __init__(self, d_model, d_ff, dropout_prob=0.1):
        """
        Args:
            d_model (int): Dimensionality of the input.
            d_ff (int): Dimensionality of the feed-forward layer.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(PositionWiseFeedForward, self).__init__()

        # Feed-forward network with two linear layers and ReLU activation.
        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the feed-forward network.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            out (Tensor): Output tensor of the same shape as input.
        """
        # Apply the first linear transformation, activation, and dropout.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Apply the second linear transformation.
        out = self.fc2(x)

        return out
    
class PositionalEncoding(nn.Module):
    """
    Positional encoding is used to inject information about the relative or absolute position of the tokens in the sequence.
    It helps the model understand the order of the tokens, as the transformer architecture does not inherently capture this information.
    """
    def __init__(self, d_model, max_seq_length=5000):
        """
        Args:
            d_model (int): Dimensionality of the model.
            max_seq_length (int): Maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model) # A tensor filled with zeros, which will be populated with positional encodings.
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # A tensor containing the position indices for each positon in the sequence.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) # A term used to scale the position indices in a spcific way.
        
        # Apply the sine function to the even indices.
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply the cosine the function to the odd indices.
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer, which means it will be part of the module's state but will not be considered a trainable parameter.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Add positional encoding to input tensor.
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder block that consists of multi-head self-attention and feed-forward layers.
    The encoder processes the input sequences and generates a contextual representation of the input.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_prob=0.3):
        """
        Transformer Encoder block that consists of multi-head self-attention and feed-forward layers.

        Args:
            d_model (int): Dimensionality of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the feed-forward layer.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(TransformerEncoder, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads) # multi-head attention mechanism.
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout_prob) # position-wise feed-forward neural network.
        self.norm1 = nn.LayerNorm(d_model) # layer normalization, applied to smooth the layer's input.
        self.norm2 = nn.LayerNorm(d_model) # layer normalization, applied to smooth the layer's input.
        self.dropout = nn.Dropout(dropout_prob) # dropout layer, used to prevent overfitting by randomly setting some activatons to zero during training.
        
    def forward(self, x, mask):
        """
        Forward pass through the transformer encoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor): Mask tensor to prevent attention to certain positions.

        Returns:
            out (Tensor): Output tensor of the same shape as input.
        """
        # Apply multi-head self-attention
        attn_output = self.self_attn(x, x, x, mask)

        # Apply dropout and layer normalization.
        x = self.norm1(x + self.dropout(attn_output))

        # Apply feed-forward network
        ff_output = self.feed_forward(x)

        # Apply dropout and layer normalization.
        out = self.norm2(x + self.dropout(ff_output))

        return out

########################################################################################################
#                                                                                                      #
#                                     QATransformerBasedModel                                          #
#                                                                                                      #
########################################################################################################
class QATransformerBasedModel(nn.Module):
    def __init__(
        self, vocab, vocab_decoder, embedding_dim, d_ff,
        num_layers, num_heads, max_seq_length, dropout_prob=0.3
    ):
        super().__init__()

        # Embedding + Positional Encoding
        self.embedding = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=embedding_dim,
            padding_idx=vocab.encoding[PAD_TOKEN]
        )
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length)

        # Split layers into pre- and post-cross-attention
        num_pre = num_layers // 2
        num_post = num_layers - num_pre
        self.pre_encoders = nn.ModuleList([
            TransformerEncoder(embedding_dim, num_heads, d_ff, dropout_prob)
            for _ in range(num_pre)
        ])
        self.post_encoders = nn.ModuleList([
            TransformerEncoder(embedding_dim, num_heads, d_ff, dropout_prob)
            for _ in range(num_post)
        ])

        # Cross-Attention
        self.cross_attn = MultiHeadAttention(embedding_dim, num_heads)
        self.cross_attn_norm = nn.LayerNorm(embedding_dim, embedding_dim)
        self.cross_attn_dropout = nn.Dropout(dropout_prob)

        # Span Prediction Head (FFN)
        self.ffn1 = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.norm1 = nn.LayerNorm(2 * embedding_dim)
        self.ffn2 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 2)
        self.dropout = nn.Dropout(dropout_prob)

        self.vocab = vocab

    def forward(self, context_ids, question_ids):
        # Embedding + Positional Encoding
        c = self.embedding(context_ids)
        c = self.pos_encoding(c)
        q = self.embedding(question_ids)
        q = self.pos_encoding(q)

        # Masks
        c_mask = (context_ids != self.vocab.encoding[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
        q_mask = (question_ids != self.vocab.encoding[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)

        # 1) Pre-cross self-attention stacks
        for enc in self.pre_encoders:
            c = enc(c, c_mask)
            q = enc(q, q_mask)

        # 2) Cross-attention: context attends to question
        c = self.cross_attn(query=c, key=q, value=q, mask=q_mask)
        c = self.cross_attn_dropout(c)
        c = self.cross_attn_norm(c)

        # 3) Post-cross self-attention stacks
        for enc in self.post_encoders:
            c = enc(c, c_mask)

        # 4) Span prediction feed-forward head
        x = self.dropout(F.relu(self.norm1(self.ffn1(c))))
        x = self.dropout(F.relu(self.norm2(self.ffn2(x))))
        logits = self.classifier(x)  # (B, L, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
    

########################################################################################################
#                                                                                                      #
#                                   QATransformerCoAttentionModel                                      #
#                                                                                                      #
########################################################################################################
class QATransformerBasedModel(nn.Module):
    def __init__(
        self, vocab, vocab_decoder, embedding_dim, d_ff,
        num_layers, num_heads, max_seq_length, dropout_prob=0.3
    ):
        super().__init__()

        self.vocab = vocab
        self.vocab_decoder = vocab_decoder
        self.dropout_prob = dropout_prob

        # Embedding map using pre-trained GloVe embeddings
        self.embedding_map = load_embeddings("./glove/glove.6B.300d.txt")

        # Initialize Embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings = len(vocab),
            embedding_dim = embedding_dim,
            padding_idx = vocab.encoding[PAD_TOKEN]
        )

        # Load pretrained weights
        embedding_matrix = create_embedding_matrix(vocab, self.embedding_map)
        self.embedding_layer.weight.data.copy_(embedding_matrix)

        # Freeze only GloVe vectors
        for i, word in enumerate(vocab.words):
            if word in self.embedding_map and word != UNK_TOKEN:
                self.embedding_layer.weight.requires_grad_(False)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length)

        # Split layers into pre- and post-cross-attention
        self.post_encoders = nn.ModuleList([
            TransformerEncoder(embedding_dim, num_heads, d_ff, dropout_prob)
            for _ in range(num_layers)
        ])

        # Cross-Attention
        self.cross_attn = MultiHeadAttention(embedding_dim, num_heads)
        self.cross_attn_norm = nn.LayerNorm(embedding_dim, embedding_dim)
        self.cross_attn_dropout = nn.Dropout(dropout_prob)

        # Span Prediction Head (FFN)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 2)
        )
        
        self.vocab = vocab

    def forward(self, context_ids, question_ids):
        # Embedding + Positional Encoding
        c = self.embedding_layer(context_ids)
        c = self.pos_encoding(c)
        q = self.embedding_layer(question_ids)
        q = self.pos_encoding(q)

        # Masks
        c_mask = (context_ids != self.vocab.encoding[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)

        # 1) Cross-attention: context attends to question
        c = self.cross_attn(query=c, key=q, value=q, mask=None)
        c = self.cross_attn_dropout(c)
        c = self.cross_attn_norm(c)

        # 2) Pre-cross self-attention stacks
        for enc in self.post_encoders:
            c = enc(c, c_mask)

        # 4) Span prediction feed-forward head
        logits = self.classifier(c)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)