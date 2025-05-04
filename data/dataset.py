import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Dict, Any, Union, Optional, Tuple
import json
class TorchDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, device):
        for k, v in self.items():
            if hasattr(v, 'to'):
                self[k] = v.to(device)
        return self

class RISCVDataset(Dataset):

    def __init__(self, h5_path: str):

        self.h5_path = h5_path

        with h5py.File(h5_path, 'r') as f:
            self.num_samples = f.attrs['num_samples']
    
    def __len__(self) -> int:

        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a sample from the dataset

        Args:
            idx: Sample index

        Returns:
            Dictionary containing features and labels
        """

        actual_idx = self.indices[idx] if hasattr(self, 'indices') else idx

        with h5py.File(self.h5_path, 'r') as f:

            X = torch.tensor(f['X'][actual_idx], dtype=torch.long)
            instruction_count = torch.tensor(f['instruction_counts'][actual_idx], dtype=torch.long)
            Y = torch.tensor(f['Y'][actual_idx], dtype=torch.float)

            instruction_text = None
            if 'instruction_text' in f:
                instruction_text = f['instruction_text'][actual_idx]

        sample = {
            'X': X,  #encoded matrix
            'instruction_count': instruction_count,
            'Y': Y
        }

        if instruction_text is not None:
            sample['instruction_text'] = instruction_text

        return sample


def make_attention_weight(mask, is_continual_pad=True):
    sizes = (~mask).sum(dim=1)
    maximum_size = mask.size(1)

    all_masking = []

    for idx, s in enumerate(sizes):
        cur_mask = ~mask[idx]

        i, j = torch.meshgrid(
            torch.arange(s, device=mask.device), torch.arange(s, device=mask.device), indexing='ij'
        )

        if is_continual_pad:
            masking = F.pad((s - abs(i - j)) / s, (0, maximum_size - s, 0, maximum_size - s), value=0)
        else:
            tmp = torch.full((maximum_size, maximum_size), 0.0, device=mask.device)
            tmp[cur_mask] = F.pad((s - abs(i - j)) / s, (0, maximum_size - s), value=0)

            masking = torch.full((maximum_size, maximum_size), 0.0, device=mask.device)
            masking[:, cur_mask] = tmp[:, :s]

        all_masking.append(masking)

    all_masking = torch.stack(all_masking)

    return all_masking

class DatasetWithDistanceWeight(Dataset):
    def __init__(self, json_path, max_instr_length = 8, max_instr_count = 64,
                 return_bb_mask=True, return_seq_mask=True, return_op_mask=True):

        self.json_path = json_path
        with open(json_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        self.pad_idx = 0
        self.return_bb_mask = return_bb_mask
        self.return_seq_mask = return_seq_mask
        self.return_op_mask = return_op_mask

        self.max_instr_length = max_instr_length
        self.max_instr_count = max_instr_count
        self.xs = self._pad_and_stack_encoded()
        self.ys = [sample['throughput'] for sample in self.data]
        self.num_instructions = [sample['num_instructions'] for sample in self.data]

    def _pad_and_stack_encoded(self):
        padded_xs = []
        for sample in self.data:
            encoded = sample['encoded']

            if len(encoded) > self.max_instr_count:
                encoded = encoded[:self.max_instr_count]

            padded_encoded = [
                F.pad(torch.tensor(instr, dtype=torch.long), (0, self.max_instr_length - len(instr)),
                      value=self.pad_idx)
                for instr in encoded
            ]

            if len(padded_encoded) < self.max_instr_count:
                padding = torch.full((self.max_instr_count - len(padded_encoded), self.max_instr_length), self.pad_idx,
                                     dtype=torch.long)
                padded_encoded.extend(padding)

            padded_xs.append(torch.stack(padded_encoded))

        return padded_xs


    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index], self.num_instructions[index]

    @staticmethod
    def make_input(x, pad_idx=0, return_bb_mask=True, return_seq_mask=True, return_op_mask=True):

        x_dict = {
            'x': x,
        }
        batch_size, inst_size, seq_size = x.shape  # batch_size, max_instr_count, max_instr_length
        mask = x == pad_idx
        bb_mask = mask.view(batch_size, inst_size * seq_size)
        seq_mask = mask.view(batch_size * inst_size, seq_size)
        op_mask = mask.all(dim=2)

        if return_bb_mask:
            # print("return_bb_mask")
            bb_attn_mod = make_attention_weight(bb_mask, is_continual_pad=False)
            x_dict['bb_attn_mod'] = bb_attn_mod

        if return_seq_mask:
            # print("return_seq_mask")
            seq_attn_mod = make_attention_weight(seq_mask)
            x_dict['seq_attn_mod'] = seq_attn_mod

        if return_op_mask:
            # print("return_op_mask")
            op_attn_mod = make_attention_weight(op_mask)
            x_dict['op_attn_mod'] = op_attn_mod

        return TorchDict(**x_dict)

def collate_fn_transformer(batch):
    xs, ys, num_instructions = zip(*batch)

    max_inst_count = max(x.size(0) for x in xs)
    max_inst_length = xs[0].size(1)

    padded_xs = []
    for x in xs:
        if x.size(0) < max_inst_count:
            padding = torch.full((max_inst_count - x.size(0), max_inst_length), 0, dtype=torch.long)
            x = torch.cat([x, padding], dim=0)
        padded_xs.append(x)

    xs = torch.stack(padded_xs)
    ys = torch.tensor(ys, dtype=torch.float)
    num_instructions = torch.tensor(num_instructions, dtype=torch.long)

    if hasattr(batch[0], '__self__'):
        dataset = batch[0].__self__
    else:
        dataset = None

    if dataset and isinstance(dataset, DatasetWithDistanceWeight):
        x_dict = DatasetWithDistanceWeight.make_input(
            xs,
            pad_idx=dataset.pad_idx,
            return_bb_mask=dataset.return_bb_mask,
            return_seq_mask=dataset.return_seq_mask,
            return_op_mask=dataset.return_op_mask
        )
    else:
        x_dict = DatasetWithDistanceWeight.make_input(xs)

    return {
        'X': x_dict,
        'instruction_count': num_instructions,
        'Y': ys
    }

class RNNDataset(Dataset):
    def __init__(self, json_path, max_instr_length = 8, max_instr_count = 64):

        self.json_path = json_path
        with open(json_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        self.pad_idx = 0
        self.max_instr_length = max_instr_length
        self.max_instr_count = max_instr_count
        self.xs = self._pad_and_stack_encoded()
        self.ys = [sample['throughput'] for sample in self.data]
        self.num_instructions = [sample['num_instructions'] for sample in self.data]

    def _pad_and_stack_encoded(self):
        padded_xs = []
        for sample in self.data:
            encoded = sample['encoded']

            if len(encoded) > self.max_instr_count:
                encoded = encoded[:self.max_instr_count]

            padded_encoded = [
                F.pad(torch.tensor(instr, dtype=torch.long), (0, self.max_instr_length - len(instr)),
                      value=self.pad_idx)
                for instr in encoded
            ]

            if len(padded_encoded) < self.max_instr_count:
                padding = torch.full((self.max_instr_count - len(padded_encoded), self.max_instr_length), self.pad_idx,
                                     dtype=torch.long)
                padded_encoded.extend(padding)

            padded_xs.append(torch.stack(padded_encoded))

        return padded_xs

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):

        sample = {
            'X':self.xs[index],
            'Y': self.ys[index],
            'instruction_count': self.num_instructions[index]
        }

        return sample


def collate_fn_lstm(batch):
    xs = [item['X'] for item in batch]
    ys = [item['Y'] for item in batch]
    instruction_counts = [item['instruction_count'] for item in batch]

    xs_batch = torch.stack(xs)
    ys_batch = torch.tensor(ys, dtype=torch.float)
    instruction_counts_batch = torch.tensor(instruction_counts, dtype=torch.long)

    return {
        'X': xs_batch,
        'Y': ys_batch,
        'instruction_count': instruction_counts_batch
    }