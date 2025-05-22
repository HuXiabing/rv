import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union, Tuple

def get_last_false_values(x, mask, dim):
    broadcast_shape = list(x.shape)
    broadcast_shape[dim] = 1

    indices = torch.argmax(mask.to(torch.int), dim=dim, keepdim=True)
    indices = indices.masked_fill(indices == 0, mask.size(dim))
    indices = indices - 1
    # indices == 0 if only it is just padding sequence or there is no padding
    #  if all padding just index does not matter

    br = torch.broadcast_to(indices.unsqueeze(-1), broadcast_shape)
    output = torch.gather(x, dim, br)
    return output.squeeze(dim)

class Fasthemal(nn.Module):
    """RISC-V instruction set throughput prediction model based on Ithemal, implemented using BatchRNN"""

    def __init__(self, config):

        super(Fasthemal, self).__init__()
        self.config = config
        self.device = torch.device(config.device)
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

    def count_parameters(self) -> int:
        print("self.config.embed_dim", self.config.embed_dim)
        print("self.config.hidden_dim", self.config.hidden_dim)
        print("self.config.num_layers", self.config.num_layers)
        print("self.config.vocab_size", self.config.vocab_size)

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class AbstractGraphModule(nn.Module):

    def __init__(self, embedding_size, hidden_size, num_classes):
        # type: (int, int, int) -> None
        super(AbstractGraphModule, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size

    def set_learnable_embedding(self, mode, dictsize, seed = None):
        # type: (str, int, Optional[int]) -> None

        self.mode = mode

        if mode != 'learnt':
            embedding = nn.Embedding(dictsize, self.embedding_size)

        if mode == 'none':
            print('learn embeddings form scratch...')
            initrange = 0.5 / self.embedding_size
            embedding.weight.data.uniform_(-initrange, initrange)
            self.final_embeddings = embedding
        elif mode == 'seed':
            print('seed by word2vec vectors....')
            embedding.weight.data = torch.FloatTensor(seed)
            self.final_embeddings = embedding
        elif mode == 'learnt':
            print('using learnt word2vec embeddings...')
            self.final_embeddings = seed
        else:
            print('embedding not selected...')
            exit()


    def load_checkpoint_file(self, fname):
        self.load_state_dict(torch.load(fname)['model'])

    def load_state_dict(self, state_dict):
        model_dict = self.state_dict()
        new_model_dict = {k: v for (k, v) in state_dict.items() if k in model_dict}
        model_dict.update(new_model_dict)
        super(AbstractGraphModule, self).load_state_dict(model_dict)

    def init_hidden(self):
        # type: () -> Tuple[nn.Parameter, nn.Parameter]

        return (
            nn.Parameter(torch.zeros(1, 1, self.hidden_size, requires_grad=True)).to(self.device),
            nn.Parameter(torch.zeros(1, 1, self.hidden_size, requires_grad=True)).to(self.device),
        )

class BatchRNN(AbstractGraphModule):
    def __init__(self, embedding_size=512, hidden_size=512, num_classes=1,
                 pad_idx=0, num_layers=1, vocab_size=256):
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
