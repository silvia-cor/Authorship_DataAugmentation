import copy
import os
import random
import sys

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer

from general.helpers_text import encod_onehot

torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

sys.setrecursionlimit(50000)


# basic class for Dataset
class CustomDataset(Dataset):
    def __init__(self, dic):
        self.samples = dic[0]
        self.labels = dic[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]


# dataloader for NN classification
class CustomDataloader:
    def __init__(self, data, feats_extractor, batch_size, shuffle=True, labels_float=True, onehot=False,
                 nn_embed_path=None):
        if nn_embed_path is not None:
            self.nn = copy.deepcopy(feats_extractor.model).to('cpu')
            self.nn.load_state_dict(torch.load(nn_embed_path))
        self.labels_float = labels_float
        self.onehot = onehot
        self.bf_extractor = feats_extractor.bf_extractor
        self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab = list(self.tokenizer.encoder.keys())
        dataset = CustomDataset(data)
        self.dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=8,
                                     worker_init_fn=self._seed_worker, collate_fn=self._collate_fn)

    # set the seed for the DataLoader worker
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _collate_fn(self, batch):
        samples = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        if self.labels_float:
            labels = torch.Tensor(labels).float()
        if not hasattr(self, 'nn'):
            final_samples = self.tokenizer(samples, return_tensors="pt", padding=True)["input_ids"].long()
            if self.onehot:
                final_samples = [encod_onehot(final_sample, self.vocab) for final_sample in final_samples]
                final_samples = torch.stack(final_samples, dim=0).float()
        else:
            with torch.no_grad():
                final_samples = [self.nn.embedding(self.tokenizer(sample, return_tensors="pt")["input_ids"])
                                 if isinstance(sample, str) else sample for sample in samples]
                max_len = max(final_samples, key=lambda x: x.shape[1]).shape[1]
                pad_vector = self.nn.embedding(torch.unsqueeze(torch.Tensor([self.tokenizer.pad_token_id]).long(), dim=0))
                final_samples = [
                    torch.cat((final_sample, pad_vector.repeat(1, (max_len - final_sample.shape[1]), 1)), dim=1)
                    for final_sample in final_samples]
            final_samples = torch.cat(final_samples, dim=0).float()
        if self.bf_extractor is not None:
            bf = torch.from_numpy(self.bf_extractor.extract(samples).todense()).float()
            return final_samples, bf, labels, samples
        else:
            return final_samples, None, labels, samples


class BertDataloader:
    def __init__(self, data, tokenizer, batch_size, labels_long=False, shuffle=True):
        self.tokenizer = tokenizer
        dataset = CustomDataset(data)
        self.dataloader = DataLoader(dataset, batch_size, num_workers=5, shuffle=shuffle,
                                     worker_init_fn=self._seed_worker, collate_fn=self._collate_fn)
        self.labels_long =labels_long

    # set the seed for the DataLoader worker
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _collate_fn(self, batch):
        documents = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        if self.labels_long:
            labels = torch.Tensor(labels).long()
        else:
            labels = torch.Tensor(labels).float()
        inputs, segs, masks = list(), list(), list()
        for text in documents:
            encoded_text = self.tokenizer.encode(text, add_special_tokens=False)
            encoded_text = torch.tensor([self.tokenizer.cls_token_id] + encoded_text + [self.tokenizer.sep_token_id])
            length = len(encoded_text)
            segmentation = torch.tensor([0] * length)
            mask = torch.tensor([1] * length)
            inputs.append(encoded_text)
            segs.append(segmentation)
            masks.append(mask)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        segs = pad_sequence(segs, batch_first=True, padding_value=0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return documents, inputs, segs, masks, labels
