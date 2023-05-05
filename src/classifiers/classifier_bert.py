import copy
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from general.dataloader import BertDataloader
from general.utils import xavier_uniform

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

batch_size = 32
epochs = 10

torch.multiprocessing.set_sharing_strategy('file_system')
sys.setrecursionlimit(50000)


class Classifier_Bert:
    def __init__(self, device):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-cased').to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.device = device

    def encode(self, data):
        dataloader = BertDataloader(data, self.tokenizer, batch_size, labels_long=False, shuffle=False).dataloader
        encod_data = []
        with torch.no_grad():
            with tqdm(dataloader, unit="batch") as data:
                self.model.eval()
                for _, inputs, segs, masks, _ in data:
                    inputs = inputs.to(self.device)
                    segs = segs.to(self.device)
                    masks = masks.to(self.device)
                    output = self.model(inputs, token_type_ids=segs, attention_mask=masks, output_hidden_states=True)
                    encod_data.extend(output.hidden_states[-1].mean(axis=1).detach().cpu())  # mean embeddings (N x 768)
        return encod_data


def Bert_classification(tr, val, te, y_tr, y_val, y_te, **kwargs):
    Cl_bert = Classifier_Bert(kwargs['device'])

    tr_dataloader = BertDataloader([tr, y_tr], Cl_bert.tokenizer, batch_size).dataloader
    val_dataloader = BertDataloader([val, y_val], Cl_bert.tokenizer, batch_size).dataloader
    te_dataloader = BertDataloader([te, y_te], Cl_bert.tokenizer, batch_size, shuffle=False).dataloader

    Cl_bert = _trainTr_Classifier(Cl_bert, tr_dataloader, val_dataloader)
    te_all_preds, te_all_targets, Cl_bert = _test_Classifier(Cl_bert, te_dataloader)
    return te_all_preds, te_all_targets, Cl_bert


# training routine for Classifier (one single epoch)
# visualize and return epoch loss (mean on the batches)
def __train(Cl_nn, dataloader, epoch):
    epoch_loss = []
    Cl_nn.model.train()
    with tqdm(dataloader, unit="batch") as train:
        for _, inputs, segs, masks, labels in train:
            inputs = inputs.to(Cl_nn.device)
            segs = segs.to(Cl_nn.device)
            masks = masks.to(Cl_nn.device)
            labels = labels.to(Cl_nn.device)
            Cl_nn.optimizer.zero_grad()
            preds = Cl_nn.model(inputs, token_type_ids=segs, attention_mask=masks).logits
            loss = Cl_nn.criterion(preds, labels)
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
        for _, inputs, segs, masks, targets in dataloader:
            inputs = inputs.to(Cl_nn.device)
            segs = segs.to(Cl_nn.device)
            masks = masks.to(Cl_nn.device)
            preds = Cl_nn.model(inputs, token_type_ids=segs, attention_mask=masks).logits
            preds = torch.argmax(preds, dim=1)
            all_preds.extend(preds.detach().clone().cpu().numpy())
            all_targets.extend(targets.numpy())
    return all_preds, all_targets


# training on the training set
def _trainTr_Classifier(Cl_nn, tr_dataloader, val_dataloader):
    trainval_dataloader = [d for dl in [tr_dataloader, val_dataloader] for d in dl]
    for epoch in range(epochs):
        Cl_nn, _ = __train(Cl_nn, trainval_dataloader, epoch)
    return copy.deepcopy(Cl_nn)


# test the Discriminator (no training)
def _test_Classifier(Cl_nn, te_dataloader):
    te_all_preds, te_all_targets = __evaluate(Cl_nn, te_dataloader)
    return te_all_preds, te_all_targets, copy.deepcopy(Cl_nn)



