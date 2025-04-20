'''
Utility functions for embedding and data processing.
'''

import nltk
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

lemmatizer = WordNetLemmatizer()
auto_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cuda(args, tensor):
    """
    Places tensor on CUDA device (by default, uses cuda:0).
    
    Returns:
        Tensor on CUDA device.
    """
    if args.use_gpu and torch:
        return tensor.cuda()
    else:
        return tensor

def unpack(tensor):
    """
    Unpacks a tensor into a Python list.

    Args:
        tensor: PyTorch tensor.

    Returns:
        Python list with tensor contents.
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy().tolist()

def load_embeddings(path):
    """
    Loads GloVe-style embeddings into memory.
    Args:
        path: Embedding path, e.g. "glove/glove.6B.300d.txt".

    Returns:
        Dictionary mapping words (strings) to vectors (list of floats).
    """
    embedding_map = {}
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                pieces = line.rstrip().split()
                word = pieces[0].lower()  # Normalize to lowercase
                embedding_map[word] = [float(weight) for weight in pieces[1:]]
                
                # Also store lemma if different
                lemma = lemmatizer.lemmatize(word)
                if lemma != word and lemma not in embedding_map:
                    embedding_map[lemma] = [float(weight) for weight in pieces[1:]]
            except:
                pass
    return embedding_map

def embed_batch(embedding_map, embedding_layer, batch_token_ids, idx2word, embed_dim):
    """
    Iteratively converts a batch of token id sequences into their embeddings.

    Args:
        embedding_map (dict): Mapping from to embedding vectors.
        batch_token_ids (List[List[int]]): Batch where each element is a list of token ids.
        idx2word (dict): Mapping from token ID (int) to the corresponding word (str)
        embed_dim (int): The dimensionality of the embeddings.
    
    Returns:
        Numpy array of shape (batch_size, seq_len, embed_dim) containing the embeddings.
    """
    batch_embeddings = []
    
    for token_ids in batch_token_ids:
        sequence_embeddings = []
        for token_id in token_ids:
            # Retrieve the corresponding word for the token id.
            token = idx2word.get(token_id.item(), None)
            # print("Token", token_id.item(), token)
            if token is None and token not in embedding_map:
                token_embedding = np.zeros(embed_dim)
            else:
                try:
                    token_tensor = torch.tensor([token_id.item()], device=device)
                    token_embedding = embedding_layer(token_tensor).squeeze(0).cpu().detach().numpy()
                except Exception as e:
                    print(f"Token ID {token_id} caused error: {e}")
                    token_embedding = np.zeros(embed_dim)

            sequence_embeddings.append(token_embedding)
        batch_embeddings.append(sequence_embeddings)
    return np.array(batch_embeddings)

def co_attention(context_embedding, question_embedding, conv=True):
    """
    Co-attention mechanism that computes attention between context and question encodings.
    If `convolution=True`, applies local smoothing to the affinity matrix.

    Args:
        context_embedding (Tensor): (B, context_len, d)
        question_embedding (Tensor): (B, question_len, d)
        convolution (bool): whether to apply convolution-based smoothing.

    Returns:
        CP (Tensor): passage attention context
        E_Out (Tensor): final encoder output
    """
    # Step 1: Affinity matrix A ∈ (B, context_len, question_len)
    A = torch.bmm(context_embedding, question_embedding.transpose(1, 2))
    # print("context_embedding = ", context_embedding[0])
    # print("question_embedding = ", question_embedding[0])
    
    # print("context_embedding:", context_embedding)
    # print("question_embedding:", question_embedding)
    # print("Affinity range:", A.min().item(), A.max().item())

    # Apply learned smoothing
    if conv:
        A = conv_co_attention(A)

    # Step 2: Passage-to-question attention (row-wise)
    A_P = F.softmax(A, dim=2)

    # Step 3: Question-to-passage attention (column-wise)
    A_Q = F.softmax(A.transpose(1, 2), dim=2)

    # Step 4: Passage attention context: CP = H^P × A^Q
    # print("Context Embedding Shape", context_embedding.shape)
    # print("Question Embedding Shape", question_embedding.shape)
    # print("A_Q Shape", A_Q.shape)
    CP = torch.bmm(A_Q, context_embedding)
    # print("CP Shape", CP.shape)  # (B, Lq, d)

    # Step 5: Encoder output: concat(H^P, [H^Q; CP] × A^P)
    # QC = torch.cat([question_embedding, CP], dim=1)
    QC_1 = torch.bmm(A_P, question_embedding)  # (B, Lq, d)
    # print("QC_1 Shape", QC_1.shape)

    QC_2 = torch.bmm(A_P, CP)  # (B, Lq, d)
    # print("QC_2 Shape", QC_2.shape)

    # add & norm
    norm_1 = nn.LayerNorm(QC_1.shape[-1]).to(QC_1.device)
    QC = norm_1(QC_1 + QC_2)  # (B, Lq, d)

    # QC = torch.cat([QC_1, QC_2], dim=1) # (B, Lq, 2d)
    # QC = torch.cat([QC_1, QC_2], dim=-1)  # (B, Lq, 2d)
    # print("QC Shape", QC.shape)

    # Final encoder output
    # E_Out = torch.cat([context_embedding, QC], dim=2)
    E_Out = torch.cat([context_embedding, QC], dim=-1)  # (B, Lq, 3d)
    E_Out = nn.LayerNorm(E_Out.shape[-1]).to(E_Out.device)(E_Out)
    E_Out = torch.tanh(E_Out)  # Apply non-linearity

    # project to original dimension
    E_Out = nn.Linear(E_Out.shape[-1], context_embedding.shape[-1]).to(E_Out.device)(E_Out)
    # print("E_Out Shape", E_Out.shape)

    return CP, E_Out

def create_gaussian_kernel(kernel_width, device, sigma=1.0):
    """Creates a 1D Gaussian kernel."""
    x = torch.arange(-kernel_width//2 + 1, kernel_width//2 + 1, dtype=torch.float, device=device)
    kernel = torch.exp(-x**2 / (2*sigma**2))
    kernel /= kernel.sum()  # Normalize to sum to 1
    return kernel.view(1, 1, -1)

def conv_co_attention(A, kernel_width=11):
    """
    Enhanced convolution to shift attention to neighboring words.
    Applies 1D convolution along context dimension per question word.
    """
    B, Lp, Lq = A.shape
    # Permute A for per-question-word processing: (B, Lq, Lp) -> (B*Lq, 1, Lp)
    A_reshaped = A.permute(0, 2, 1).reshape(-1, 1, Lp)
    
    # Create Gaussian kernel with odd kernel width (e.g., 11)
    kernel = create_gaussian_kernel(kernel_width, A.device, sigma=1.0)
    
    # Use symmetric padding that keeps the sequence length unchanged.
    padded_length = (kernel_width - 1) // 2
    smoothed_A = F.conv1d(A_reshaped, kernel, padding=padded_length)
    
    # Reshape back: current shape is (B*Lq, 1, Lp) --> (B, Lq, Lp) then permute to (B, Lp, Lq)
    smoothed_A = smoothed_A.view(B, Lq, Lp).permute(0, 2, 1)
    A_adjusted = A + smoothed_A  # Enhance original scores with neighbor context
    return F.softmax(A_adjusted, dim=-1)

def tokenize_with_bert(text):
    # Tokenize the text and request offset mappings.
    encoding = auto_tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False  # Disable adding special tokens to mimic simple whitespace tokenization.
    )
    
    # Retrieve the tokens.
    tokens = auto_tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    
    # Retrieve the spans from the offset mapping.
    spans = encoding['offset_mapping']
    return tokens, spans
    
def create_embedding_matrix(vocab, embedding_map, embedding_dim=300, scale=0.6):
    """Initialize embedding matrix with:
    - GloVe vectors for known words
    - Random vectors for UNK tokens
    - Zero vector for padding
    """
    # Initialize with random normal distribution (match GloVe scale)
    embedding_matrix = np.random.normal(
        scale=scale, 
        size=(len(vocab), embedding_dim)
    )
    
    # Handle special tokens
    embedding_matrix[vocab.encoding[PAD_TOKEN]] = np.zeros(embedding_dim)
    unk_idx = vocab.encoding[UNK_TOKEN]
    embedding_matrix[unk_idx] = np.random.normal(scale=scale, size=embedding_dim)
    
    for word, idx in vocab.encoding.items():
        if word in [PAD_TOKEN, UNK_TOKEN]:
            continue
            
        # Try direct match
        if word in embedding_map:
            embedding_matrix[idx] = embedding_map[word]
            continue
            
        # Try lemma
        lemma = lemmatizer.lemmatize(word)
        if lemma in embedding_map:
            embedding_matrix[idx] = embedding_map[lemma]
            continue
            
        # Try lowercase lemma
        lower_lemma = lemmatizer.lemmatize(word.lower())
        if lower_lemma in embedding_map:
            embedding_matrix[idx] = embedding_map[lower_lemma]

    return torch.tensor(embedding_matrix, dtype=torch.float32)

def enforce_position_constraints(end_logits, start_positions):
    """
    Mask end_logits positions before the corresponding start_positions.
    """
    batch_size, seq_len = end_logits.size()
    positions = torch.arange(seq_len, device=end_logits.device).unsqueeze(0).expand(batch_size, seq_len)
    mask = positions < start_positions.unsqueeze(1)
    return end_logits.masked_fill(mask, float('-inf'))

def span_loss_no_mask(start_logits, end_logits, start_pos, end_pos):
    # 1) start loss over full distribution
    start_loss = F.cross_entropy(start_logits, start_pos)
    
    # 2) predicted start (detach so no grad through argmax)
    with torch.no_grad():
        s_pred = start_logits.argmax(dim=1)  # (B,)
    
    # 3) end loss over full distribution (unconstrained)
    end_loss = F.cross_entropy(end_logits, end_pos)
    
    return start_loss, end_loss, s_pred, end_logits