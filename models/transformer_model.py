import torch
from .base_model import BaseModel
from .DeepPM import DeepPM

class TransformerModel(BaseModel):
    """
    基于DeepPM的RISC-V指令集吞吐量预测模型
    """

    def __init__(self, config):
        """
        初始化Transformer模型

        Args:
            config: 配置对象
        """
        super(TransformerModel, self).__init__(config)

        self.model = DeepPM(
            dim=config.embed_dim,
            n_heads=config.num_heads,
            dim_ff=config.hidden_dim,
            pad_idx=0,
            vocab_size=config.vocab_size,
            num_basic_block_layer=2,
            num_instruction_layer=2,
            num_op_layer=config.num_layers,
            use_checkpoint=getattr(config, 'use_checkpoint', False),
            use_layernorm=getattr(config, 'use_layernorm', True),
            use_bb_attn=getattr(config, 'use_bb_attn', True),
            use_seq_attn=getattr(config, 'use_seq_attn', True),
            use_op_attn=getattr(config, 'use_op_attn', True),
            use_pos_2d=getattr(config, 'use_pos_2d', False),
            dropout=config.dropout,
            pred_drop=config.dropout,
            activation='gelu',
            handle_neg=getattr(config, 'handle_neg', False)
        )

    def forward(self, x, instruction_count=None):
        """
        Args:
            x: 输入数据 [batch_size, max_instr_count, max_instr_length]
            instruction_count: 每个样本的指令数量 [batch_size]
        Returns:
            预测的吞吐量值 [batch_size]
        """

        mask = x == 0
        # 创建指令掩码
        if instruction_count is not None:
            instr_mask = torch.arange(x.size(1), device=x.device)[None, :] >= instruction_count[:, None]

            # 将超出指令数量的部分全部掩码
            for i in range(x.size(0)):  # 遍历批次
                if instruction_count[i] < x.size(1):
                    mask[i, instruction_count[i]:, :] = True

        # 准备模型输入
        model_input = {
            'x': x,
            'bb_attn_mod': None,  # 可选的注意力修饰符
            'seq_attn_mod': None,
            'op_attn_mod': None
        }

        output = self.model(model_input)

        return output