import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .base_model import BaseModel
from .Ithemal import BatchRNN, get_last_false_values, AbstractGraphModule

class Fasthemal(BaseModel):
    """RISC-V instruction set throughput prediction model based on Ithemal, implemented using BatchRNN"""

    def __init__(self, config):

        super(Fasthemal, self).__init__(config)

        # create BatchRnn model
        self.model = BatchRNN(
            embedding_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_classes=1,  # for regression task
            pad_idx=0,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size
        )

    def forward(self, x, instruction_count=None):
        """
        Args:
            x: input data [batch_size, max_instr_count, max_instr_length]
            instruction_count:  [batch_size]

        Returns:
            y: [batch_size]
        """
        # BatchRNN is already designed to handle batched inputs
        # So x can be passed directly
        return self.model(x)


class BatchRNN(AbstractGraphModule):
    def __init__(self, embedding_size=512, hidden_size=512, num_classes=1,
                 pad_idx=0, num_layers=1, vocab_size=700):
        super(BatchRNN, self).__init__(embedding_size, hidden_size, num_classes)

        self.pad_idx = pad_idx
        self.token_rnn = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, num_layers=num_layers)
        self.instr_rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, num_layers=num_layers)
        self.token_rnn.to(self.device)
        self.instr_rnn.to(self.device)

        self._token_init = self.rnn_init_hidden()
        self._instr_init = self.rnn_init_hidden()

        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        self.set_learnable_embedding(mode='none', dictsize=vocab_size)

    def rnn_init_hidden(self):
        # type: () -> Union[Tuple[nn.Parameter, nn.Parameter], nn.Parameter]

        hidden = self.init_hidden()

        # for h in hidden:
        #     torch.nn.init.kaiming_uniform_(h)

        return hidden
        # if self.params.rnn_type == RnnType.LSTM:
        #     return hidden
        # else:
        #     return hidden[0]

    def get_token_init(self):
        # type: () -> torch.tensor
        # if self.params.learn_init:
        #     return self._token_init
        # else:
        return self.rnn_init_hidden()

    def get_instr_init(self):
        # type: () -> torch.tensor
        # if self.params.learn_init:
        #     return self._instr_init
        # else:
        return self.rnn_init_hidden()

    def pred_of_instr_chain(self, instr_chain):
        # type: (torch.tensor) -> torch.tensor
        _, final_state_packed = self.instr_rnn(instr_chain, self.get_instr_init())
        final_state = final_state_packed[0]
        # if self.params.rnn_type == RnnType.LSTM:
        #     final_state = final_state_packed[0]
        # else:
        #     final_state = final_state_packed
        return self.linear(final_state.squeeze()).squeeze()

    def forward(self, x):
        # mask = B I S
        # x = B I S

        mask = x == self.pad_idx

        batch_size, inst_size, seq_size = x.shape

        #  tokens = B I S HID
        tokens = self.final_embeddings(x)

        #  B*I S HID
        tokens = tokens.view(batch_size * inst_size, seq_size, -1)

        # output = B*I S HID
        output, _ = self.token_rnn(tokens)

        #  B I S HID
        output = output.view(batch_size, inst_size, seq_size, -1)

        #  B I HID
        instr_chain = get_last_false_values(output, mask, dim=2)

        #  B I HID
        inst_output, _ = self.instr_rnn(instr_chain)

        #  B I
        mask = mask.all(dim=-1)

        #  B HID
        final_state = get_last_false_values(inst_output, mask, dim=1)

        #  B
        output = self.linear(final_state).squeeze(-1)
        return output
