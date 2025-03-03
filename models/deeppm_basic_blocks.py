import torch
import torch.nn as nn
from .deeppm_transformer import DeepPMTransformerEncoderLayer
from torch.utils.checkpoint import checkpoint
from .checkpoint_utils import method_dummy_wrapper

def get_device(should_print=True):
    if torch.cuda.is_available():
        str_device = 'cuda'
    else:
        str_device = 'cpu'

    device = torch.device(str_device)

    if should_print:
        print(f'Using {device}')
    return device

class DeepPMSeq(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, dropout=None, 
                use_layernorm=False, layer_norm_eps=1e-05, use_checkpoint=False, activation='gelu', dummy=None,
                handle_neg=False):
        super().__init__()

        self.tr = nn.ModuleList(
            [
                DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff,
                        use_layernorm=use_layernorm, layer_norm_eps=layer_norm_eps,
                        dropout=dropout, activation=activation, handle_neg=handle_neg)
                    for _ in range(n_layers)
            ]
        )

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            if dummy is None:
                device = get_device(should_print=False)
                self.dummy = torch.zeros(1, requires_grad=True, device=device)
            else:
                self.dummy = dummy
    
    def forward(self, x, mask, op_seq_mask, weighted_attn=None):
        """
        x: [batch_size, inst_size, seq_size, dim]
        mask: [batch_size, inst_size, seq_size]
        op_seq_mask: [batch_size, inst_size]
        """
        
        batch_size, inst_size, seq_size, _ = x.shape

        x = x.view(batch_size * inst_size, seq_size, -1)  # (batch_size * inst_size, seq_size, dim)
        mask = mask.view(batch_size * inst_size, seq_size)  # (batch_size * inst_size, seq_size)
        op_seq_mask = op_seq_mask.view(batch_size * inst_size)  # (batch_size * inst_size)

        x = x.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)  # (batch_size * inst_size, seq_size, dim)
        mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)  # (batch_size * inst_size, seq_size)

        for block in self.tr:
            if self.use_checkpoint:
                x = checkpoint(method_dummy_wrapper(block), self.dummy, x, mod_mask, weighted_attn)
            else:
                x = block(x, mod_mask, weighted_attn)

        x = x.masked_fill(mask.unsqueeze(-1), 0)  # (batch_size * inst_size, seq_size, dim)
        x = x.view(batch_size, inst_size, seq_size, -1)  # (batch_size, inst_size, seq_size, dim)
        return x
    
class DeepPMBasicBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, dropout=None, 
                use_layernorm=False, layer_norm_eps=1e-05, use_checkpoint=False, activation='gelu', dummy=None,
                handle_neg=False):
        super().__init__()

        self.tr = nn.ModuleList(
            [
                DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff,
                        use_layernorm=use_layernorm, layer_norm_eps=layer_norm_eps,
                        dropout=dropout, activation=activation, handle_neg=handle_neg)
                    for _ in range(n_layers)
            ]
        )

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            if dummy is None:
                device = get_device(should_print=False)
                self.dummy = torch.zeros(1, requires_grad=True, device=device)
            else:
                self.dummy = dummy
        
    def forward(self, x, mask, weighted_attn=None):
        """
        x: [batch_size, inst_size, seq_size, dim]
        mask: [batch_size, inst_size, seq_size]
        """
        batch_size, inst_size, seq_size, _ = x.shape  # (batch_size, inst_size, seq_size, dim)

        x = x.view(batch_size, inst_size * seq_size, -1)  # (batch_size, inst_size * seq_size, dim)
        mask = mask.view(batch_size, inst_size * seq_size)  # (batch_size, inst_size * seq_size)

        for block in self.tr:
            if self.use_checkpoint:
                x = checkpoint(method_dummy_wrapper(block), self.dummy, x, mask, weighted_attn)
            else:
                x = block(x, mask, weighted_attn)

        x = x.masked_fill(mask.unsqueeze(-1), 0)  # (batch_size, inst_size * seq_size, dim)
        x = x.view(batch_size, inst_size, seq_size, -1)  # (batch_size, inst_size, seq_size, dim)
        return x
    
class DeepPMOp(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, dropout=None, 
                use_layernorm=False, layer_norm_eps=1e-05, use_checkpoint=False, activation='gelu', dummy=None,
                handle_neg=False):
        super().__init__()

        self.tr = nn.ModuleList(
            [
                DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff,
                        use_layernorm=use_layernorm, layer_norm_eps=layer_norm_eps,
                        dropout=dropout, activation=activation, handle_neg=handle_neg)
                    for _ in range(n_layers)
            ]
        )

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            if dummy is None:
                device = get_device(should_print=False)
                self.dummy = torch.zeros(1, requires_grad=True, device=device)
            else:
                self.dummy = dummy

    def forward(self, x, op_seq_mask, weighted_attn=None):
        """
        x: [batch_size, inst_size, seq_size, dim]
        op_seq_mask: [batch_size, inst_size]
        """
        for block in self.tr:
            if self.use_checkpoint:
                x = checkpoint(method_dummy_wrapper(block), self.dummy, x, op_seq_mask, weighted_attn)
            else:
                x = block(x, op_seq_mask, weighted_attn)

        x = x.masked_fill(op_seq_mask.unsqueeze(-1), 0)  
        return x
    