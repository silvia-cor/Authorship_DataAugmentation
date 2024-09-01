import os
import random

import numpy as np
import torch
from torch import nn
from transformers import GPT2Tokenizer

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


class Generator_embed_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = 128
        self.layer_size = 512


class Generator_embed_model_gru(Generator_embed_model):
    def __init__(self):
        super().__init__()
        num_layers_gru = 2
        self.gru = nn.GRU(input_size=self.emb_dim, hidden_size=self.layer_size, num_layers=num_layers_gru,
                          batch_first=True, bidirectional=False)
        self.dense = nn.Linear(self.layer_size, self.emb_dim)

    def forward(self, x):
        return forward_gru(self, x, dense=False)


class Generator_embed_model_trans(Generator_embed_model):
    def __init__(self):
        super().__init__()
        num_heads = 4
        self.positional_encoder = PositionalEncoding(d_model=self.emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=num_heads,
                                                   dim_feedforward=self.layer_size)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dense = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, x):
        return forward_trans(self, x, dense=False)

# class for all the components of a generator (Gru or Transformer) processing embeddings
class Generator_embed:
    def __init__(self, G_path, device):
        self.model = Generator_embed_model_gru() if "Gru" in G_path else Generator_embed_model_trans()
        self.model = self.model.to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.device = device
        self.LM_criterion = nn.CosineEmbeddingLoss().to(device)
        self.gan_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.9))
        #torch.optim.RMSprop(self.model.parameters(), lr=5e-5)

    def generate(self, author_samples, **kwargs):
        if isinstance(author_samples, list):
            l = len(author_samples)
        else:
            l = author_samples.size(dim=0)
        author_sample = author_samples[np.random.choice(l)]
        if isinstance(author_sample, str):
            author_sample = self.tokenizer.encode(author_sample, return_tensors="pt")[0]
            with torch.no_grad():
                generated_sample = kwargs['Cl_nn'].model.embedding(
                    torch.unsqueeze(author_sample[:5], dim=0).to(self.device))
        else:
            generated_sample = torch.unsqueeze(author_sample[:5], dim=0).to(self.device)
        while generated_sample.shape[1] < len(author_sample):
            next_token = self.model.forward(generated_sample)
            generated_sample = torch.cat((generated_sample, next_token), dim=1)
        generated_sample_det = generated_sample.detach().clone().cpu()
        return generated_sample, generated_sample_det

    def LM_train(self, inputs, targets):
        targets = targets.to(self.device)
        inputs = torch.unsqueeze(inputs, dim=0).to(self.device)
        preds = torch.squeeze(self.model(inputs))
        sim_target = torch.tensor(1).to(self.device)
        loss = self.LM_criterion(preds, targets, sim_target)
        return loss

    def prepare_for_LMtraining(self, texts):
        lm_inputs, lm_targets = get_examples_for_LM(texts, self.tokenizer)
        return lm_inputs, lm_targets
