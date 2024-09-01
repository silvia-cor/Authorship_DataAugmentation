import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
from tqdm import tqdm

from classifiers.classifier_nn import Classifier_NN
from general.dataloader import CustomDataloader
from general.helpers_text import delete_gpt2tokenizer_unwanted_symbols, encod_onehot
from generators.generator_embed import Generator_embed
from generators.generator_gpt2 import Generator_gpt2
from generators.generator_onehot import Generator_onehot
from generators.helpers_gen import apply_loss, penalize_grad

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

torch.multiprocessing.set_sharing_strategy('file_system')
# batch size params
batch_size_gan = 8  # batch size for the GAN training
batch_size_lm = 32  # batch size for the LM training
# number of epochs params
n_epochs_GANTuning_gpt2 = 10
n_epochs_GANTuning_OnehotEmbed = 500
n_epochs_GANTuning_D = 5  # epochs for training the discriminator
n_epochs_LM_gpt2 = 3
n_epochs_LM_OnehotEmbed = 300
# number of generated sequences param
return_sequences_ratio = 10  # ratio of generated sequences relative to the original number of chunks per author (e.g., now it's 10 times the original number of chunks)


def TuningGeneration_pipeline(tuning_policy, author_samples, author, G_path, nn_path, device, gan_device):
    generated_samples = []
    n = len(author_samples) * return_sequences_ratio
    num_return_sequences = n if n < 1000 else 1000
    if "gpt2" in G_path:
        G = Generator_gpt2(device)
        if tuning_policy == "LMT":
            G = _LMTuning_gpt2(G, author_samples, author, G_path)
        else:
            Cl_nn = Classifier_NN(emb_layer=True, first_layer_dim=768, tr=None, y_tr=None, device=device,
                                  gan_device=gan_device)
            G = _GANTraining(G, Cl_nn, author_samples, author, G_path, onehot=False, nn_path=None,
                             n_epochs=n_epochs_GANTuning_gpt2)
        print(f"Generating {num_return_sequences} new samples...")
        for i in tqdm(range(num_return_sequences)):
            _, _, _, generated_text = G.generate(author_samples, check_tokens=True)
            generated_samples.append(delete_gpt2tokenizer_unwanted_symbols(generated_text))
    elif "onehot" in G_path:
        G = Generator_onehot(G_path, device)
        if tuning_policy == "LMT":
            G = _LMTraining_OnehotEmbed(G, author_samples, author, G_path, None)
        else:
            Cl_nn = Classifier_NN(emb_layer=False, first_layer_dim=128, tr=None, y_tr=None, device=device,
                                  gan_device=gan_device)
            G = _GANTraining(G, Cl_nn, author_samples, author, G_path, onehot=True, nn_path=None,
                             n_epochs=n_epochs_GANTuning_OnehotEmbed)
        print(f"Generating {num_return_sequences} new samples...")
        for i in tqdm(range(num_return_sequences)):
            _, _, generated_text = G.generate(author_samples)
            generated_samples.append(delete_gpt2tokenizer_unwanted_symbols(generated_text))
    else:
        G = Generator_embed(G_path, device)
        Cl_nn = Classifier_NN(emb_layer=True, first_layer_dim=128, tr=None, y_tr=None, device=device,
                              gan_device=gan_device)
        if tuning_policy == "LMT":
            G = _LMTraining_OnehotEmbed(G, author_samples, author, G_path, Cl_nn)
        else:
            G = _GANTraining(G, Cl_nn, author_samples, author, G_path, onehot=False, nn_path=nn_path,
                             n_epochs=n_epochs_GANTuning_OnehotEmbed)
        print(f"Generating {num_return_sequences} new samples...")
        for i in tqdm(range(num_return_sequences)):
            _, generated_embed = G.generate(author_samples, Cl_nn=Cl_nn)
            generated_samples.append(generated_embed)
    return generated_samples


def _LMTuning_gpt2(G, author_samples, author, G_path):
    os.makedirs(str(Path(G_path).parent), exist_ok=True)
    if os.path.isfile(G_path):
        print(f'LM-tuned generator for {author} found!')
        G.model.load_state_dict(torch.load(G_path, map_location=G.device))
    else:
        print('LM-tuning the generator...')
        G.model.train()
        for epoch in range(n_epochs_LM_gpt2):
            epoch_loss = []
            for i in (train := tqdm(range(len(author_samples)))):
                loss = G.fine_tune(author_samples[i]).loss
                loss_item, G.optimizer = apply_loss(G.optimizer, loss, batch_size_lm, i, len(author_samples))
                epoch_loss.append(loss_item)
                train.set_description(f'Epoch {epoch + 1} loss={np.mean(epoch_loss):.5f}')
        torch.save(G.model.state_dict(), G_path)
    return G


def _LMTraining_OnehotEmbed(G, author_samples, author, G_path, Cl_nn):
    os.makedirs(str(Path(G_path).parent), exist_ok=True)
    if os.path.isfile(G_path):
        print(f'LM-trained generator for {author} found!')
        G.model.load_state_dict(torch.load(G_path, map_location=G.device))
    else:
        print('LM-training the generator...')
        G.model.train()
        inputs, targets = G.prepare_for_LMtraining(author_samples)
        for epoch in range(n_epochs_LM_OnehotEmbed):
            epoch_loss = []
            for i in (train := tqdm(range(len(inputs)))):
                if Cl_nn is None:
                    inp = torch.Tensor(encod_onehot(inputs[i], G.vocab)).float()
                    target = torch.Tensor(encod_onehot(targets[i], G.vocab)).float()
                else:
                    with torch.no_grad():
                        inp = Cl_nn.model.embedding(inputs[i].to(G.device)).detach().clone().cpu().float()
                        target = Cl_nn.model.embedding(targets[i].to(G.device)).detach().clone().cpu().float()
                loss = G.LM_train(inp, target)
                loss_item, G.optimizer = apply_loss(G.optimizer, loss, batch_size_lm, i, len(inputs))
                epoch_loss.append(loss_item)
                train.set_description(f'Epoch {epoch + 1} loss={np.mean(epoch_loss):.5f}')
        torch.save(G.model.state_dict(), G_path)
    return G


def _GANTraining(G, Cl_nn, author_samples, author, G_path, onehot, nn_path, n_epochs):
    if os.path.isfile(G_path):
        print(f'GAN-trained generator for {author} found!')
        G.model.load_state_dict(torch.load(G_path, map_location=G.device))
    else:
        #one = torch.tensor(1, dtype=torch.float).to(Cl_nn.device)
        #mone = one * -1
        dataloader_real = CustomDataloader([author_samples, [1] * len(author_samples)], Cl_nn, batch_size_gan,
                                           onehot=onehot, nn_embed_path=nn_path).dataloader
        for epoch in range(n_epochs):
            print(f'GAN epoch {epoch + 1} for author {author}')
            # (1) Update D network n times
            print('Training Discriminator...')
            for i in range(n_epochs_GANTuning_D):
                __GAN_train_discriminator(Cl_nn, G, dataloader_real)  #, mone, one)
            # (2) Update G network
            print('Training Generator...')
            __GAN_train_generator(Cl_nn, G, dataloader_real)  #, mone)
        torch.save(G.model.state_dict(), G_path)
        G.model.eval()
    return G


def __GAN_train_discriminator(D, G, dataloader_real):  #, mone, one):
    D.model.train()
    G.model.eval()
    losses = []
    with tqdm(dataloader_real, unit='batch') as train:
        for inputs, _, _, _ in train:
            #  for p in D.model.parameters():
            #    p.data.clamp_(-0.01, 0.01)
            D.gan_optimizer.zero_grad()
            # train with real
            inputs = inputs.to(D.device)
            d_real = D.model(inputs, None)
            #d_real = d_real.mean()
            #d_real.backward(mone)
            # train with fake
            fakes = [G.generate(inputs, Cl_nn=D, check_tokens=False)[1] for i in range(inputs.shape[0])]
            fakes = torch.cat(fakes, dim=0).to(D.device)
            if len(inputs.shape) == 2:
                inputs = torch.stack([G.embedding(inp).detach().clone() for inp in inputs], dim=0).to(D.device)
            d_fake = D.model(fakes, None)
            #d_fake = d_fake.mean()
            #d_fake.backward(one)
            # grad penalty
            grad_penalty = penalize_grad(D, inputs, fakes)
            #grad_penalty.backward()
            d_loss = -torch.mean(d_real) + torch.mean(d_fake) + grad_penalty
            #losses.append(d_fake.detach().clone().cpu() - d_real.detach().clone().cpu() +
            #              grad_penalty.detach().clone().cpu())
            losses.append(d_loss.detach().clone().cpu())
            d_loss.backward()
            D.gan_optimizer.step()
            train.set_description(f'Discriminator loss={np.mean(losses):.5f}')


def __GAN_train_generator(D, G, dataloader_real):  #, mone):
    D.model.eval()
    G.model.train()
    losses = []
    with tqdm(dataloader_real, unit='batch') as train:
        for inputs, _, _, _ in train:
            G.gan_optimizer.zero_grad()
            fakes = [G.generate(inputs, Cl_nn=D, check_tokens=False)[1] for i in range(inputs.shape[0])]
            fakes = torch.cat(fakes, dim=0).float().to('cpu')
            d_model = torch.nn.DataParallel(D.model, device_ids=D.gan_device)
            #g = g.mean()
            #g.backward(mone)
            d_fake = d_model(fakes, None)
            g_loss = -torch.mean(d_fake)
            g_loss.backward()
            losses.append(g_loss.detach().clone().cpu())
            G.gan_optimizer.step()
            train.set_description(f'Generator loss={np.mean(losses):.5f}')
