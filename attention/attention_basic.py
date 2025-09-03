import torch.nn.functional as F
import torch
import math

def scaled_dot_product_attention(query, key, value, mask = None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attention_weights, value)

    return output, attention_weights

def generate_causal_mask(seq_len, device = 'CPU'):
    mask = torch \
        .tril(torch.ones(seq_len, seq_len, device=device)) \
        .view(1, 1, seq_len, seq_len)
    
    return mask

batch_size, seq_len, d_model = 2, 5, 64
num_heads = 8
head_dim = d_model // num_heads

Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
K = torch.randn(batch_size, num_heads, seq_len, head_dim)
V = torch.randn(batch_size, num_heads, seq_len, head_dim)

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Output: {output}")
print(f"Attention weights: {attn_weights}")

causal_mask = generate_causal_mask(seq_len, Q.device)
causal_output, causal_attn_weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print(f"Masked output: {causal_output}")
print(f"Masked attention weights: {causal_attn_weights}")