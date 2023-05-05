import copy
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

from BaseFeatures_extractor import featuresExtractor
from general.dataloader import CustomDataloader
from general.utils import xavier_uniform

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

torch.multiprocessing.set_sharing_strategy('file_system')
sys.setrecursionlimit(50000)

batch_size = 32
patience = 25
max_epochs = 500
min_epochs = 49


class NN_model(nn.Module):
    def __init__(self, vocab_len, first_layer_dim):
        super().__init__()
        self.vocab_len = vocab_len
        self.kernel_sizes = [3, 5]
        self.first_layer_dim = first_layer_dim
        self.dropout = nn.Dropout(0.3)
        self.maxpool = nn.MaxPool1d(3)
        self.flat = nn.Flatten()
        self.cnn1 = nn.ModuleList(
            [nn.Conv1d(self.first_layer_dim, 512, kernel_size) for kernel_size in self.kernel_sizes])
        self.cnn2 = nn.ModuleList([nn.Conv1d(512, 256, kernel_size) for kernel_size in self.kernel_sizes])
        self.dense1 = nn.Linear(256 * len(self.kernel_sizes), 64)

    def representation(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass

    def _conv_block(self, x, conv_layers):
        x = x.transpose(1, 2).contiguous()  # (N, Cin, L)
        for i, conv_layer in enumerate(conv_layers):
            x = F.relu(conv_layer(x))  # (N, Cout, L)
            if i < len(conv_layers) - 1:
                x = self.maxpool(x)
        L = x.size()[2]
        x = F.max_pool1d(x, L)  # (N, Cout, 1)
        x = x.squeeze(2)  # (N, Cout)
        return x  # output (N, Cout)


# for all cases with gpt2, and _embed (which does not have bf)
class NN_with_embeddingLayer(NN_model):
    def __init__(self, vocab_len, emb_dim, bf_dim):
        super().__init__(vocab_len, emb_dim)
        self.emb_dim = emb_dim
        if bf_dim is not None:
            self.bf_layer1 = nn.Linear(bf_dim, 128)
            self.bf_layer2 = nn.Linear(128, 64)
            self.dense2 = nn.Linear(64 * 2, 2)
        else:
            self.dense2 = nn.Linear(64, 2)
        self.embed = nn.Embedding(self.vocab_len, self.emb_dim)

    def representation(self, x, bf):
        if len(x.shape) == 2:  # if x is list of ids
            x = self.embedding(x)
        conv_stack = [list(convs) for convs in zip(self.cnn1, self.cnn2)]
        x = [self._conv_block(x, conv_group) for conv_group in conv_stack]
        x = torch.cat(x, dim=1)
        x = self.flat(x)
        x = self.dropout(x)
        x = F.relu(self.dense1(x))
        if bf is not None:
            bf = F.relu(self.bf_layer1(bf))
            bf = F.relu(self.bf_layer2(bf))
            x = torch.cat((bf, x), dim=1)
        return x

    def forward(self, x, bf):
        x = self.representation(x, bf)
        x = self.dense2(x)
        return x

    def embedding(self, x):
        return self.embed(x)

    def emb_requires_grad(self, requires_grad):
        for param in self.embed.parameters():
            param.requires_grad = requires_grad


# for _onehot
class NN_without_embeddingLayer(NN_model):
    def __init__(self, vocab_len, emb_dense_dim, bf_dim):
        super().__init__(vocab_len, emb_dense_dim)
        self.emb_dense_dim = emb_dense_dim
        self.bf_layer1 = nn.Linear(bf_dim, 128)
        self.bf_layer2 = nn.Linear(128, 64)
        self.emb_dense = nn.Linear(self.vocab_len, self.emb_dense_dim, bias=False)
        self.dense2 = nn.Linear(64 * 2, 2)

    def representation(self, x, bf):
        bf = F.relu(self.bf_layer1(bf))
        bf = F.relu(self.bf_layer2(bf))
        conv_stack = [list(convs) for convs in zip(self.cnn1, self.cnn2)]
        x = self.emb_dense(x)
        x = [self._conv_block(x, conv_group) for conv_group in conv_stack]
        x = torch.cat(x, dim=1)
        x = self.flat(x)
        x = self.dropout(x)
        x = F.relu(self.dense1(x))
        x = torch.cat((bf, x), dim=1)
        return x

    def forward(self, x, bf):
        x = self.representation(x, bf)
        x = self.dense2(x)
        return x


class Classifier_NN:
    def __init__(self, emb_layer, first_layer_dim, tr, y_tr, device, vocab_len=50257):
        if tr:
            self.bf_extractor = featuresExtractor(tr)
            self.bf_dim = self.bf_extractor.extract(tr).shape[1]
        else:
            self.bf_dim = None
            self.bf_extractor = None
        if emb_layer:
            self.model = NN_with_embeddingLayer(vocab_len, first_layer_dim, self.bf_dim).to(device)
        else:
            self.model = NN_without_embeddingLayer(vocab_len, first_layer_dim, self.bf_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        class_weights = compute_class_weight('balanced', classes=np.unique(np.array(y_tr)), y=np.array(y_tr))
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean').to(device)
        self.device = device


def NN_classification(tr, val, te, y_tr, y_val, y_te, device, generator_name, nn_base_path, nn_final_path,
                      nn_embed_path):
    onehot = True if "onehot" in nn_base_path else False
    if generator_name == "gpt2":
        Cl_nn = Classifier_NN(emb_layer=True, first_layer_dim=768, tr=tr, y_tr=y_tr, device=device)
    elif "onehot" in generator_name:
        Cl_nn = Classifier_NN(emb_layer=False, first_layer_dim=128, tr=tr, y_tr=y_tr, device=device)
    else:
        Cl_nn = Classifier_NN(emb_layer=True, first_layer_dim=128, tr=None, y_tr=y_tr, device=device)

    tr_dataloader = CustomDataloader([tr, y_tr], Cl_nn, batch_size,
                                     onehot=onehot, nn_embed_path=nn_embed_path).dataloader
    val_dataloader = CustomDataloader([val, y_val], Cl_nn, batch_size,
                                      onehot=onehot, nn_embed_path=nn_embed_path).dataloader
    te_dataloader = CustomDataloader([te, y_te], Cl_nn, batch_size, shuffle=False,
                                     onehot=onehot, nn_embed_path=nn_embed_path).dataloader

    val_f1s = _trainTr_Classifier(Cl_nn, tr_dataloader, val_dataloader, nn_base_path)
    _trainVal_Classifier(Cl_nn, tr_dataloader, val_dataloader, nn_base_path, nn_final_path)
    te_all_preds, te_all_targets, Cl_nn = _test_Classifier(Cl_nn, te_dataloader, nn_final_path)
    return te_all_preds, te_all_targets, Cl_nn


# training routine for Classifier (one single epoch)
# visualize and return epoch loss (mean on the batches)
def __train(Cl_nn, dataloader, epoch):
    epoch_loss = []
    Cl_nn.model.train()
    with tqdm(dataloader, unit="batch") as train:
        for inputs, bf, targets in train:
            Cl_nn.optimizer.zero_grad()
            inputs = inputs.to(Cl_nn.device)
            if bf is not None:
                bf = bf.to(Cl_nn.device)
            targets = targets.to(Cl_nn.device)
            preds = Cl_nn.model(inputs, bf)
            loss = Cl_nn.criterion(preds, targets)
            loss.backward()
            Cl_nn.optimizer.step()
            epoch_loss.append(loss.item())
            train.set_description(f'Epoch {epoch + 1} loss={np.mean(epoch_loss):.5f}')
    return copy.deepcopy(Cl_nn), np.mean(epoch_loss)


# evaluation routine for Classifier (no training)
# returns preds and targets for the entire dataloader
def __evaluate(Cl_nn, dataloader):
    Cl_nn.model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for inputs, bf, targets in dataloader:
            inputs = inputs.to(Cl_nn.device)
            if bf is not None:
                bf = bf.to(Cl_nn.device)
            preds = Cl_nn.model(inputs, bf)
            preds = torch.argmax(preds, dim=1)
            all_preds.extend(preds.detach().clone().cpu().numpy())
            all_targets.extend(targets.numpy())
    return all_preds, all_targets


# training on the training set
def _trainTr_Classifier(Cl_nn, tr_dataloader, val_dataloader, nn_base_path):
    xavier_uniform(Cl_nn.model)
    val_f1s = []
    epochs_no_improv = 0
    val_f1_epoch, val_f1_max = 0, 0
    torch.save(Cl_nn.model.state_dict(), nn_base_path)
    for epoch in range(max_epochs):
        model, _ = __train(Cl_nn, tr_dataloader, epoch)
        val_all_preds, val_all_targets = __evaluate(model, val_dataloader)
        val_f1_epoch = f1_score(val_all_targets, val_all_preds, average='binary', zero_division=1)
        # if after patience there is no improvement, early stop happens
        # for the first min_epochs, no early stopping anyway
        if val_f1_epoch > val_f1_max:
            epochs_no_improv = 0
            torch.save(Cl_nn.model.state_dict(), nn_base_path)
            val_f1_max = val_f1_epoch
        else:
            if epoch > min_epochs:
                epochs_no_improv += 1
        val_f1s.append(val_f1_epoch)
        print(f'Val_F1_max: {val_f1_max:.3f} Val_F1_epoch: {val_f1_epoch:.3f}')
        if epochs_no_improv == patience and epoch > min_epochs:
            print("Early stopping!")
            break
    return val_f1s


# final training on the validation set
def _trainVal_Classifier(Cl_nn, tr_dataloader, val_dataloader, nn_base_path, nn_final_path):
    Cl_nn.model.load_state_dict(torch.load(nn_base_path))
    trainval_dataloader = [d for dl in [tr_dataloader, val_dataloader] for d in dl]
    for epoch in range(5):
        Cl_nn, _ = __train(Cl_nn, trainval_dataloader, epoch)
    torch.save(Cl_nn.model.state_dict(), nn_final_path)


# test the Discriminator (no training)
def _test_Classifier(Cl_nn, te_dataloader, nn_path):
    Cl_nn.model.load_state_dict(torch.load(nn_path))
    te_all_preds, te_all_targets = __evaluate(Cl_nn, te_dataloader)
    return te_all_preds, te_all_targets, copy.deepcopy(Cl_nn)
