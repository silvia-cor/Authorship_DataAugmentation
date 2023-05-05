import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
from torch import nn
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

from generators.generator_gpt2 import Generator_gpt2
from generators.generator_onehot import Generator_onehot
from generators.generator_embed import Generator_embed
from classifiers.classifier_nn import Classifier_NN, __train
from general.dataloader import CustomDataloader
from general.helpers_text import delete_gpt2tokenizer_unwanted_symbols, tokenize_nopunct
from generators.helpers_gen import compute_loss

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

torch.multiprocessing.set_sharing_strategy('file_system')
batch_size = 32
n_epochs_GANTuning_gpt2 = 10
n_epochs_GANTuning_OnehotEmbed = 100
n_epochs_GANTuning_D = 5
n_epochs_LM_gpt2 = 5
n_epochs_LM_OnehotEmbed = 50
return_sequences_ratio = 5


def TuningGeneration_pipeline(tuning_policy, tr, y_tr, author_samples, author, n_negatives, G_path, nn_base_path, device):
    class_weights = compute_class_weight('balanced', classes=np.unique(np.array(y_tr)), y=np.array(y_tr))
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    gan_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean').to(device)
    generated_samples = []
    n = len(author_samples) * return_sequences_ratio
    num_return_sequences = n if n < 500 else 500
    num_final_return_sequences = num_return_sequences
    # num_final_return_sequences = n_negatives // 3
    if "gpt2" in G_path:
        G = Generator_gpt2(device)
        if tuning_policy == "LMT":
            G = _LMTuning_gpt2(G, author_samples, author, G_path)
        else:
            Cl_nn = Classifier_NN(emb_layer=True, first_layer_dim=768, tr=tr, y_tr=y_tr, device=device)
            Cl_nn.model.load_state_dict(torch.load(nn_base_path, map_location=Cl_nn.device))
            G = _GANTuning_gpt2(G, Cl_nn, author_samples, author, num_return_sequences, G_path, gan_criterion, device)
        print(f"Generating {num_final_return_sequences} new samples...")
        while len(generated_samples) < num_final_return_sequences:
            generated_text, _ = G.generate(author_samples)
            if len(tokenize_nopunct(generated_text)) >= 25:
                generated_samples.append(delete_gpt2tokenizer_unwanted_symbols(generated_text))
    elif "onehot" in G_path:
        G = Generator_onehot(G_path, device)
        if tuning_policy == "LMT":
            G = _LMTraining_OnehotEmbed(G, author_samples, author, G_path, None)
        else:
            Cl_nn = Classifier_NN(emb_layer=False, first_layer_dim=128, tr=tr, y_tr=y_tr, device=device)
            Cl_nn.model.load_state_dict(torch.load(nn_base_path, map_location=Cl_nn.device))
            G = _GANTraining_onehot(G, Cl_nn, author_samples, author, num_return_sequences, G_path, gan_criterion, device)
        print(f"Generating {num_final_return_sequences} new samples...")
        for i in tqdm(range(num_final_return_sequences)):
            _, _, generated_text = G.generate(author_samples)
            generated_samples.append(delete_gpt2tokenizer_unwanted_symbols(generated_text))
    else:
        G = Generator_embed(G_path, device)
        Cl_nn = Classifier_NN(emb_layer=True, first_layer_dim=128, tr=None, y_tr=y_tr, device=device)
        Cl_nn.model.load_state_dict(torch.load(nn_base_path, map_location=Cl_nn.device))
        if tuning_policy == "LMT":
            G = _LMTraining_OnehotEmbed(G, author_samples, author, G_path, Cl_nn)
        else:
            G = _GANTraining_embed(G, Cl_nn, author_samples, author, num_return_sequences, G_path, gan_criterion, device)
        print(f"Generating {num_final_return_sequences} new samples...")
        for i in tqdm(range(num_final_return_sequences)):
            _, generated_embed = G.generate(author_samples, Cl_nn)
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
                loss_item, G.optimizer = compute_loss(G.optimizer, loss, batch_size, i, len(author_samples))
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
        if Cl_nn is None:
            inputs, targets = G.prepare_for_LMtraining(author_samples)
        else:
            inputs, targets = G.prepare_for_LMtraining(author_samples, Cl_nn)
        for epoch in range(n_epochs_LM_OnehotEmbed):
            epoch_loss = []
            for i in (train := tqdm(range(len(inputs)))):
                loss = G.LM_train(inputs[i], targets[i])
                loss_item, G.optimizer = compute_loss(G.optimizer, loss, batch_size, i, len(inputs))
                epoch_loss.append(loss_item)
                train.set_description(f'Epoch {epoch + 1} loss={np.mean(epoch_loss):.5f}')
        torch.save(G.model.state_dict(), G_path)
    return G


def _GANTuning_gpt2(G, Cl_nn, author_samples, author, num_return_sequences, G_path, criterion, device):
    G.model.train()
    if os.path.isfile(G_path):
        print(f'GAN-tuned generator for {author} found!')
        G.model.load_state_dict(torch.load(G_path, map_location=G.device))
    else:
        for epoch in range(n_epochs_GANTuning_gpt2):
            print(f'GAN epoch {epoch + 1} for author {author}')
            Cl_nn.model.eval()
            losses = []
            generated_texts = []
            generated_encods = []
            print('Creating samples...')
            while len(generated_texts) < num_return_sequences:
                generated_text, generated_encod = G.generate(author_samples)
                if len(tokenize_nopunct(generated_text)) >= 25:
                    generated_texts.append(generated_text)
                    generated_encods.append(generated_encod)
            print('Tuning the Generator...')
            for i in tqdm(range(len(generated_texts))):
                bf = torch.from_numpy(Cl_nn.bf_extractor.extract([generated_texts[i]]).todense()).float()
                generated_embedding = G.embedding(
                    torch.unsqueeze(generated_encods[i], dim=0))  # the embedding has a gradient attached
                preds = Cl_nn.model(generated_embedding, bf.to(device))
                targets = torch.Tensor([1]).long().to(device)
                loss = criterion(preds, targets)
                loss_item, G.optimizer = compute_loss(G.optimizer, loss, batch_size, i, len(generated_texts))
                losses.append(loss_item)
            epoch_loss = np.mean(losses)
            print(f'G_loss: {epoch_loss:.5f}')
            texts = author_samples + generated_texts
            labels = [1] * len(author_samples) + [0] * len(generated_texts)
            gan_dataloader = CustomDataloader([texts, labels], Cl_nn, batch_size).dataloader
            if epoch < n_epochs_GANTuning_gpt2 - 1:
                print('Training the Discriminator...')
                # train D
                for D_epoch in range(n_epochs_GANTuning_D):
                    Cl_nn, _ = __train(Cl_nn, gan_dataloader, D_epoch)
        torch.save(G.model.state_dict(), G_path)
    return G


def _GANTraining_onehot(G, Cl_nn, author_samples, author, num_return_sequences, G_path, criterion, device):
    G.model.train()
    if os.path.isfile(G_path):
        print(f'GAN-trained generator for {author} found!')
        G.model.load_state_dict(torch.load(G_path, map_location=G.device))
    else:
        for epoch in range(n_epochs_GANTuning_OnehotEmbed):
            print(f'GAN epoch {epoch + 1} for author {author}')
            Cl_nn.model.eval()
            losses = []
            generated_texts = []
            print('Creating samples and training the Generator...')
            for i in tqdm(range(num_return_sequences)):
                generated_onehot, _, generated_text = G.generate(author_samples)
                bf = torch.from_numpy(Cl_nn.bf_extractor.extract([generated_text]).todense()).float()
                preds = Cl_nn.model(generated_onehot, bf.to(device))
                targets = torch.Tensor([1]).long().to(device)
                loss = criterion(preds, targets)
                loss_item, G.optimizer = compute_loss(G.optimizer, loss, batch_size, i, num_return_sequences)
                losses.append(loss_item)
                generated_texts.append(generated_text)
            epoch_loss = np.mean(losses)
            print(f'G_loss: {epoch_loss:.5f}')
            texts = author_samples + generated_texts
            labels = [1] * len(author_samples) + [0] * len(generated_texts)
            gan_dataloader = CustomDataloader([texts, labels], Cl_nn, batch_size, onehot=True).dataloader
            if epoch < n_epochs_GANTuning_OnehotEmbed - 1:
                print('Training the Discriminator...')
                # train D
                for D_epoch in range(n_epochs_GANTuning_D):
                    Cl_nn, _ = __train(Cl_nn, gan_dataloader, D_epoch)
        torch.save(G.model.state_dict(), G_path)
    return G


def _GANTraining_embed(G, Cl_nn, author_samples, author, num_return_sequences, G_path, criterion, device):
    G.model.train()
    if os.path.isfile(G_path):
        print(f'GAN-trained generator for {author} found!')
        G.model.load_state_dict(torch.load(G_path, map_location=G.device))
    else:
        for epoch in range(n_epochs_GANTuning_OnehotEmbed):
            print(f'GAN epoch {epoch + 1} for author {author}')
            Cl_nn.model.eval()
            losses = []
            generated_embeds = []
            print('Creating samples and training the Generator...')
            for i in tqdm(range(num_return_sequences)):
                generated_embed, generated_embed_det = G.generate(author_samples, Cl_nn)
                preds = Cl_nn.model(generated_embed, bf=None)
                targets = torch.Tensor([1]).long().to(device)
                loss = criterion(preds, targets)
                loss_item, G.optimizer = compute_loss(G.optimizer, loss, batch_size, i, num_return_sequences)
                losses.append(loss_item)
                generated_embeds.append(generated_embed_det)
            epoch_loss = np.mean(losses)
            print(f'G_loss: {epoch_loss:.5f}')
            texts = author_samples + generated_embeds
            labels = [1] * len(author_samples) + [0] * len(generated_embeds)
            gan_dataloader = CustomDataloader([texts, labels], Cl_nn, batch_size).dataloader
            if epoch < n_epochs_GANTuning_OnehotEmbed - 1:
                print('Training the Discriminator...')
                # train D
                for D_epoch in range(n_epochs_GANTuning_D):
                    Cl_nn, _ = __train(Cl_nn, gan_dataloader, D_epoch)
        torch.save(G.model.state_dict(), G_path)
    return G
