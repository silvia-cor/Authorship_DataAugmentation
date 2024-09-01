import os
import random

import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed

from general.helpers_text import tokenize_nopunct

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
set_seed(42)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

torch.multiprocessing.set_sharing_strategy('file_system')

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token


# class for all the components of the GPT2 generator
class Generator_gpt2:
    def __init__(self, device):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('distilgpt2',
                                                     pad_token_id=tokenizer.eos_token_id).to(device)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.device = device
        self.gan_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.9))
        #torch.optim.RMSprop(self.model.parameters(), lr=5e-5)

    def fine_tune(self, text):
        tokenized_text = tokenizer(text, return_tensors="pt")
        inp = tokenized_text.to(self.device)
        targets = tokenized_text["input_ids"].to(self.device)
        outputs = self.model(**inp, labels=targets)
        return outputs

    def embedding(self, encod):
        encod = encod.to(self.device)
        embed = self.model(encod, output_hidden_states=True).hidden_states[-1]
        return embed

    def generate(self, author_samples, **kwargs):
        if isinstance(author_samples, list):
            l = len(author_samples)
        else:
            l = author_samples.size(dim=0)
        author_sample = author_samples[np.random.choice(range(l))]
        if isinstance(author_sample, str):
            author_sample = tokenizer.encode(author_sample, return_tensors="pt")[0]
        inp = torch.unsqueeze(author_sample[:5], dim=0).to(self.device)
        generated_encod = self.model.generate(inp, min_length=len(author_sample), max_length=len(author_sample),
                                              do_sample=True, top_k=50, no_repeat_ngram_size=5)[0]
        generated_text = tokenizer.decode(generated_encod, skip_special_tokens=True)
        generated_embed = torch.unsqueeze(self.embedding(generated_encod), dim=0)
        generated_embed_det = generated_embed.detach().clone().cpu()
        if kwargs['check_tokens']:
            if len(tokenize_nopunct(generated_text)) < 25:
                del generated_embed, generated_embed_det, generated_encod, generated_text
                generated_embed, generated_embed_det, generated_encod, generated_text = self.generate(author_samples,
                                                                                                      check_tokens=True)
        return generated_embed, generated_embed_det, generated_encod, generated_text
