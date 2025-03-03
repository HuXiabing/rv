import torch
import torch.nn as nn

from .pos_encoder import get_positional_encoding_1d, get_positional_encoding_2d
from .deeppm_basic_blocks import DeepPMBasicBlock, DeepPMSeq, DeepPMOp

def get_device(should_print=True):
    if torch.cuda.is_available():
        str_device = 'cuda'
    else:
        str_device = 'cpu'

    device = torch.device(str_device)

    if should_print:
        print(f'Using {device}')
    return device

class DeepPM(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=0, vocab_size=700,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, use_checkpoint=False, use_layernorm=False,
                use_bb_attn=True, use_seq_attn=True, use_op_attn=True,
                use_pos_2d=False, dropout=None, pred_drop=0.0, activation='gelu', handle_neg=False):
        super().__init__()

        self.num_basic_block_layer = num_basic_block_layer
        self.num_instruction_layer = num_instruction_layer
        self.num_op_layer = num_op_layer
        if self.num_basic_block_layer <= 0:
            raise ValueError('num_basic_block_layer must be larger than 1')

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            device = get_device(should_print = True)
            self.dummy = torch.zeros(1, requires_grad=True, device=device)
        else:
            self.dummy = None

        self.use_pos_2d = use_pos_2d
        if self.use_pos_2d:
            self.pos_embed_2d = get_positional_encoding_2d(dim)
        
        self.pos_embed = get_positional_encoding_1d(dim)


        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, self.pad_idx)


        self.basic_block = DeepPMBasicBlock(dim, dim_ff, n_heads, num_basic_block_layer, use_layernorm=use_layernorm,
                            use_checkpoint=use_checkpoint, dummy=self.dummy, dropout=dropout, activation=activation,
                            handle_neg=handle_neg)
        
        if self.num_instruction_layer > 0:
            self.instruction_block = DeepPMSeq(dim, dim_ff, n_heads, num_instruction_layer, use_layernorm=use_layernorm,
                            use_checkpoint=use_checkpoint, dummy=self.dummy, dropout=dropout, activation=activation, 
                            handle_neg=handle_neg)
        if self.num_op_layer > 0:
            self.op_block = DeepPMOp(dim, dim_ff, n_heads, num_op_layer, use_layernorm=use_layernorm,
                            use_checkpoint=use_checkpoint, dummy=self.dummy, dropout=dropout, activation=activation,
                            handle_neg=handle_neg)

        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1)
        )

        self.use_bb_attn = use_bb_attn
        self.use_seq_attn = use_seq_attn
        self.use_op_attn = use_op_attn
       
    def forward(self, x):
        bb_attn_mod = x['bb_attn_mod'] if self.use_bb_attn else None
        seq_attn_mod = x['seq_attn_mod'] if self.use_seq_attn else None
        op_attn_mod = x['op_attn_mod'] if self.use_op_attn else None
        x = x['x']

        # B I S
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)

        #  B I S D
        output = self.embed(x)

        if self.use_pos_2d:
            output = self.pos_embed_2d(output)
        else:
            # B*I S H
            output = output.view(batch_size * inst_size, seq_size, -1)
            output = self.pos_embed(output)

            # B I S H
            output = output.view(batch_size, inst_size, seq_size, -1)

        output = self.basic_block(output, mask, bb_attn_mod)

        if self.num_instruction_layer > 0:
            output = self.instruction_block(output, mask, op_seq_mask, seq_attn_mod)

        # reduce
        # B I H
        output = output[:, :, 0]
        output = self.pos_embed(output)

        if self.num_op_layer > 0:
            output = self.op_block(output, op_seq_mask, op_attn_mod)

        #  B I
        output = output.sum(dim=1)
        output = self.prediction(output).squeeze(1)
        return output
