import random

import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
set_seed(42)

torch.multiprocessing.set_sharing_strategy('file_system')

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token


class Generator_gpt2:
    def __init__(self, device):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('distilgpt2',
                                                     pad_token_id=tokenizer.eos_token_id).to(device)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.device = device

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

    def generate(self, author_samples):
        author_sample = tokenizer.encode(np.random.choice(author_samples), return_tensors="pt")[0]
        inp = torch.unsqueeze(author_sample[:5], dim=0).to(self.device)
        generated_encod = self.model.generate(inp, min_length=len(author_sample), max_length=len(author_sample),
                                              do_sample=True, top_k=50, no_repeat_ngram_size=5)[0]
        generated_text = tokenizer.decode(generated_encod, skip_special_tokens=True)
        return generated_text, generated_encod
