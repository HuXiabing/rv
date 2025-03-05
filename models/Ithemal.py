from enum import Enum, unique
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union, Tuple

def get_device(should_print=True):
    if torch.cuda.is_available():
        str_device = 'cuda'
    else:
        str_device = 'cpu'

    device = torch.device(str_device)

    if should_print:
        print(f'Using {device}')
    return device

class AbstractGraphModule(nn.Module):

    def __init__(self, embedding_size, hidden_size, num_classes):
        # type: (int, int, int) -> None
        super(AbstractGraphModule, self).__init__()

        self.device = get_device(False)
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

    def remove_refs(self, item):
        # type: (dt.DataItem) -> None
        pass

class ModelAbs(nn.Module):

    """
    Abstract model without the forward method.

    lstm for processing tokens in sequence and linear layer for output generation
    lstm is a uni-directional single layer lstm

    num_classes = 1 - for regression
    num_classes = n - for classifying into n classes

    """

    def __init__(self, hidden_size, embedding_size, num_classes):

        super(ModelAbs, self).__init__()
        self.hidden_size = hidden_size
        self.name = 'should be overridden'

        #numpy array with batchsize, embedding_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        #lstm - input size, hidden size, num layers
        self.lstm_token = nn.LSTM(self.embedding_size, self.hidden_size)

        #hidden state for the rnn
        self.hidden_token = self.init_hidden()

        #linear layer for regression - in_features, out_features
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)))


    #this is to set learnable embeddings
    def set_learnable_embedding(self, mode, dictsize, seed = None):

        self.mode = mode

        if mode != 'learnt':
            embedding = nn.Embedding(dictsize, self.embedding_size)

        if mode == 'none':
            print( 'learn embeddings form scratch...')
            initrange = 0.5 / self.embedding_size
            embedding.weight.data.uniform_(-initrange, initrange)
            self.final_embeddings = embedding
        elif mode == 'seed':
            print( 'seed by word2vec vectors....')
            embedding.weight.data = torch.FloatTensor(seed)
            self.final_embeddings = embedding
        else:
            print( 'using learnt word2vec embeddings...')
            self.final_embeddings = seed

    #remove any references you may have that inhibits garbage collection
    def remove_refs(self, item):
        return

class ModelHierarchicalRNN(ModelAbs):

    """
    Prediction at every hidden state of the unrolled rnn for instructions.

    Input - sequence of tokens processed in sequence by the lstm but seperated into instructions
    Output - predictions at the every hidden state

    lstm predicting instruction embedding for sequence of tokens
    lstm_ins processes sequence of instruction embeddings
    linear layer process hidden states to produce output

    """

    def __init__(self, hidden_size, embedding_size, num_classes, intermediate):
        super(ModelHierarchicalRNN, self).__init__(hidden_size, embedding_size, num_classes)

        self.hidden_ins = self.init_hidden()
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)

        if intermediate:
            self.name = 'hierarchical RNN intermediate'
        else:
            self.name = 'hierarchical RNN'
        self.intermediate = intermediate

    def copy(self, model):

        self.linear = model.linear
        self.lstm_token = model.lstm_token
        self.lstm_ins = model.lstm_ins

    def forward(self, item):

        self.hidden_token = self.init_hidden()
        self.hidden_ins = self.init_hidden()

        ins_embeds = autograd.Variable(torch.zeros(len(item.x),self.embedding_size))
        for i, ins in enumerate(item.x):

            if self.mode == 'learnt':
                acc_embeds = []
                for token in ins:
                    acc_embeds.append(self.final_embeddings[token])
                token_embeds = torch.FloatTensor(acc_embeds)
            else:
                token_embeds = self.final_embeddings(torch.LongTensor(ins))

            #token_embeds = torch.FloatTensor(ins)
            token_embeds_lstm = token_embeds.unsqueeze(1)
            out_token, hidden_token = self.lstm_token(token_embeds_lstm,self.hidden_token)
            ins_embeds[i] = hidden_token[0].squeeze()

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        out_ins, hidden_ins = self.lstm_ins(ins_embeds_lstm, self.hidden_ins)

        if self.intermediate:
            values = self.linear(out_ins[:,0,:]).squeeze()
        else:
            values = self.linear(hidden_ins[0].squeeze()).squeeze()

        return values


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
