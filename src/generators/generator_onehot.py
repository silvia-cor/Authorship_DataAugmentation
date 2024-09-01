import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Tokenizer

from general.helpers_text import encod_onehot, onehot_text
from general.utils import PositionalEncoding
from generators.helpers_gen import forward_trans, forward_gru, get_examples_for_LM

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

torch.multiprocessing.set_sharing_strategy('file_system')


# note: lstm input is : input, (h_0, c_0)
# if (h_0, c_0) is not provided, they are set to default to zeros
# hence, there is no need for 'manual' initialization
class Generator_onehot_model(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab_len = len(vocab)
        self.dense_emb_dim = 128
        self.dense_emb = nn.Linear(self.vocab_len, self.dense_emb_dim, bias=False)
        self.layer_size = 512


class Generator_onehot_model_gru(Generator_onehot_model):
    def __init__(self, vocab):
        super().__init__(vocab)
        num_layers_gru = 2
        self.gru = nn.GRU(input_size=self.dense_emb_dim, hidden_size=self.layer_size, num_layers=num_layers_gru,
                          batch_first=True, bidirectional=False)
        self.dense = nn.Linear(self.layer_size, self.vocab_len)

    def forward(self, x):
        return forward_gru(self, x, dense=True)


class Generator_onehot_model_trans(Generator_onehot_model):
    def __init__(self, vocab):
        super().__init__(vocab)
        num_heads = 4
        self.positional_encoder = PositionalEncoding(d_model=self.dense_emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dense_emb_dim, nhead=num_heads,
                                                   dim_feedforward=self.layer_size)  # default= 8
        # output is the same shape of the input (d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # default= 6
        self.dense = nn.Linear(self.dense_emb_dim, self.vocab_len)

    def forward(self, x):
        return forward_trans(self, x, dense=True)


# class for all the components of a generator (Gru or Transformer) processing onehot vectors
class Generator_onehot:
    def __init__(self, G_path, device):
        self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab = list(self.tokenizer.encoder.keys())
        self.model = Generator_onehot_model_gru(self.vocab) if "Gru" in G_path \
            else Generator_onehot_model_trans(self.vocab)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.device = device
        self.LM_criterion = nn.CrossEntropyLoss().to(device)
        self.gan_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.9))
        #self.gan_optimizer = torch.optim.RMSprop(self.model.parameters(), lr=5e-5)

    def generate(self, author_samples, **kwargs):
        if isinstance(author_samples, list):
            l = len(author_samples)
        else:
            l = author_samples.size(dim=0)
        author_sample = author_samples[np.random.choice(l)]
        if isinstance(author_sample, str):
            author_sample = encod_onehot(self.tokenizer.encode(author_sample, return_tensors="pt")[0], vocab=self.vocab)
        try:
            generated_onehot = torch.unsqueeze(author_sample[:5].float(), dim=0).to(self.device)
            while generated_onehot.shape[1] < len(author_sample):
                next_token = F.gumbel_softmax(self.model.forward(generated_onehot), hard=True)
                generated_onehot = torch.cat((generated_onehot, next_token), dim=1)
        except:
            g_device = self.device
            self.model.to('cpu')
            generated_onehot = torch.unsqueeze(author_sample[:5].float(), dim=0)
            while generated_onehot.shape[1] < len(author_sample):
                next_token = F.gumbel_softmax(self.model.forward(generated_onehot), hard=True)
                generated_onehot = torch.cat((generated_onehot, next_token), dim=1)
            self.model.to(g_device)
        generated_onehot_det = generated_onehot.detach().clone().cpu()
        generated_text = onehot_text(torch.squeeze(generated_onehot_det, dim=0), self.vocab)
        return generated_onehot, generated_onehot_det, generated_text

    def LM_train(self, inputs, targets):
        targets = targets.to(self.device)
        inputs = torch.unsqueeze(inputs, dim=0).to(self.device)
        preds = torch.squeeze(self.model(inputs))
        loss = self.LM_criterion(preds, targets)
        return loss

    def prepare_for_LMtraining(self, texts):
        lm_inputs, lm_targets = get_examples_for_LM(texts, self.tokenizer)
        return lm_inputs, lm_targets
