import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .checkpoint_utils import method_dummy_wrapper
from .CustomSelfAttention import CustomSelfAttention

def get_device(should_print=True):
    if torch.cuda.is_available():
        str_device = 'cuda'
    else:
        str_device = 'cpu'

    device = torch.device(str_device)

    if should_print:
        print(f'Using {device}')
    return device

class DeePPMTransformerEncoder(nn.Module):
    def __init__(self, num_layers, dim, n_heads, dim_ff=2048, use_layernorm=False, 
                 layer_norm_eps=1e-05, dropout=None, use_checkpoint=False, activation='gelu', handle_neg=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff,
                                use_layernorm=use_layernorm, layer_norm_eps=layer_norm_eps, dropout=dropout,
                                activation=activation, handle_neg=handle_neg) 
                    for _ in range(num_layers)
            ]
        )

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            device = get_device(should_print=False)
            self.dummy = torch.zeros(1, requires_grad=True, device=device)

    def forward(self, src, src_key_padding_mask=None, weighted_attn=None):
        for block in self.layers:
            if self.use_checkpoint:
                output = checkpoint(method_dummy_wrapper(block), self.dummy, src, src_key_padding_mask, weighted_attn)
            else:
                output = block(src, src_key_padding_mask, weighted_attn)
        return output
    
class DeepPMTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, dim_ff=2048, use_layernorm=False, layer_norm_eps=1e-05, dropout=None,
                 activation='gelu', handle_neg=False):
        super().__init__()

        if activation == 'gelu':
            act = nn.GELU
        elif activation == 'relu':
            act = nn.ReLU
        else:
            raise NotImplementedError()

        if dropout is None:
            dropout = 0.0

        self.attn = CustomSelfAttention(dim, n_heads, dropout, handle_neg=handle_neg)

        self.dropout = nn.Dropout(dropout)

        self.pwff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            act(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
            nn.Dropout(dropout)
        )

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)

    def forward(self, src, src_key_padding_mask=None, weighted_attn=None):
        x = src
        
        h = self.attn(x, key_padding_mask=src_key_padding_mask, attn_mask_modifier=weighted_attn)
        h = self.dropout(h)

        if self.use_layernorm:
            h = self.norm1(x + h)
            forward = self.pwff(h)
            h = self.norm2(h + forward)
        else:
            h = x + h
            forward = self.pwff(h)
            h = h + forward

        return h
