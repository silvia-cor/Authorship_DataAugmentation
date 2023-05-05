import os
import pickle
from pathlib import Path

import torch
import math
import torch.nn as nn
import pandas as pd


# pickle the output of a function
def pickled_resource(pickle_path: str, generation_func: callable, *args, **kwargs):
    if pickle_path is None:
        return generation_func(*args, **kwargs)
    else:
        if os.path.exists(pickle_path):
            return pickle.load(open(pickle_path, 'rb'))
        else:
            instance = generation_func(*args, **kwargs)
            os.makedirs(str(Path(pickle_path).parent), exist_ok=True)
            pickle.dump(instance, open(pickle_path, 'wb'), pickle.HIGHEST_PROTOCOL)
            return instance


def getBack(var_grad_fn):  # requires loss.grad_fn
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                # print('Tensor with grad found:', tensor)
                # print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


def xavier_uniform(model: nn.Module):
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            nn.init.xavier_uniform_(p)


def get_maxlen(a):
    return len(max(a, key=len))


def load_files(result_path, pickle_path):
    if os.path.exists(result_path):
        df_csv = pd.read_csv(result_path, sep=';')
    else:
        os.makedirs(str(Path(result_path).parent), exist_ok=True)
        columns = ['Method', 'Author_name', 'F1', 'Acc']
        df_csv = pd.DataFrame(columns=columns)
        df_csv.to_csv(path_or_buf=result_path, sep=';', index=False, header=True)
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as pickle_file:
            df_preds = pickle.load(pickle_file)
    else:
        os.makedirs(str(Path(pickle_path).parent), exist_ok=True)
        df_preds = {}
    return df_csv, df_preds


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
