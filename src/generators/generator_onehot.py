import torch
from torch import nn
import torch.nn.functional as F
from general.helpers_text import encod_onehot, onehot_text
import random
import numpy as np
from general.utils import PositionalEncoding
from transformers import GPT2Tokenizer
from generators.helpers_gen import forward_trans, forward_gru, get_examples_for_LM

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

torch.multiprocessing.set_sharing_strategy('file_system')

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token
vocab = list(tokenizer.encoder.keys())


# note: lstm input is : input, (h_0, c_0)
# if (h_0, c_0) is not provided, they are set to default to zeros
# hence, there is no need for 'manual' initialization
class Generator_onehot_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_len = len(vocab)
        self.dense_emb_dim = 128
        self.dense_emb = nn.Linear(self.vocab_len, self.dense_emb_dim, bias=False)
        self.layer_size = 512


class Generator_onehot_model_gru(Generator_onehot_model):
    def __init__(self):
        super().__init__()
        num_layers_gru = 2
        self.gru = nn.GRU(input_size=self.dense_emb_dim, hidden_size=self.layer_size, num_layers=num_layers_gru,
                          batch_first=True, bidirectional=False)
        self.dense = nn.Linear(self.layer_size, self.vocab_len)

    def forward(self, x):
        return forward_gru(self, x, dense=True)


class Generator_onehot_model_trans(Generator_onehot_model):
    def __init__(self):
        super().__init__()
        num_heads = 4
        self.positional_encoder = PositionalEncoding(d_model=self.dense_emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dense_emb_dim, nhead=num_heads,
                                                   dim_feedforward=self.layer_size)  # default= 8
        # output is the same shape of the input (d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # default= 6
        self.dense = nn.Linear(self.dense_emb_dim, self.vocab_len)

    def forward(self, x):
        return forward_trans(self, x, dense=True)


class Generator_onehot:
    def __init__(self, G_path, device):
        self.model = Generator_onehot_model_gru() if "Gru" in G_path \
            else Generator_onehot_model_trans()
        self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.device = device
        self.finetune_criterion = nn.CrossEntropyLoss().to(device)
        self.vocab = vocab

    def generate(self, author_samples):
        author_sample = tokenizer.encode(np.random.choice(author_samples), return_tensors="pt")[0]
        generated_onehot = torch.unsqueeze(encod_onehot(author_sample[:5], vocab).float(), dim=0).to(self.device)
        while generated_onehot.shape[1] < len(author_sample):
            next_token = F.gumbel_softmax(self.model.forward(generated_onehot), hard=True)
            generated_onehot = torch.cat((generated_onehot, next_token), dim=1)
        generated_onehot_det = generated_onehot.detach().clone().cpu()
        generated_text = onehot_text(torch.squeeze(generated_onehot_det, dim=0), vocab)
        return generated_onehot, generated_onehot_det, generated_text

    def LM_train(self, inputs, targets):
        targets = targets.to(self.device)
        inputs = torch.unsqueeze(inputs, dim=0).to(self.device)
        preds = torch.squeeze(self.model(inputs))
        loss = self.finetune_criterion(preds, targets)
        return loss

    def prepare_for_LMtraining(self, texts):
        lm_inputs, lm_targets = get_examples_for_LM(texts, tokenizer)
        onehot_inputs = [torch.Tensor(encod_onehot(lm_input, vocab)).float() for lm_input in lm_inputs]
        onehot_targets = [torch.Tensor(encod_onehot(lm_target, vocab)).float() for lm_target in lm_targets]
        return onehot_inputs, onehot_targets
