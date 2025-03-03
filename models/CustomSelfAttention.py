# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

# Copying code from deeppm_original_transformer.py
# which is from above

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomSelfAttention(nn.Module):
    """
    CustomSelfAttention(dim, n_heads, dropout, handle_neg=handle_neg)
    """
    def __init__(self, dim, n_heads, dropout=None, handle_neg=False):
        super().__init__()
        
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        if dropout is None:
            dropout = 0.0
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dim, dim)

        self.handle_neg = handle_neg

    def forward(self, x, key_padding_mask=None, attn_mask_modifier=None):
        # B S D
        batch_size, seq_size, _ = x.shape

        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        
        # B S H W -trans-> B H S W
        q = q.view(batch_size, seq_size, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_size, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_size, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # B H S S
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.head_dim**0.5)

        if attn_mask_modifier is not None:
            if self.handle_neg:
                energy = energy - abs(energy * (1 - attn_mask_modifier.unsqueeze(1)))
            else:
                energy = energy * attn_mask_modifier.unsqueeze(1)

        if key_padding_mask is not None:
            energy = energy.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(-1), -1e10)

        attention = F.softmax(energy, dim=-1)

        # B H S W -> B S H W
        x = torch.matmul(self.dropout(attention), v).permute(0, 2, 1, 3).contiguous()

        # B S D
        x = x.view(batch_size, seq_size, -1)

        return self.output(x)