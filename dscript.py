import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    def __init__(self, ids1, ids2, labels, embedding_h5):
        self.ids1 = ids1
        self.ids2 = ids2
        self.labels = labels
        self.embed_data = {}

        ids = set(ids1).union(set(ids2))
        with h5py.File(embedding_h5, "r") as h5fin:
            for id in ids:
                self.embed_data[id] = h5fin[id][:, :]

    def __len__(self):
        return len(self.ids1)

    def __getitem__(self, i):
        x1 = self.loader(self.ids1[i])
        x2 = self.loader(self.ids2[i])
        return x1, x2, torch.as_tensor(self.labels[i]).float()

    def loader(self, id, max_len=600):
        embedding = self.embed_data[id]
        seq_len = embedding.shape[0]
        seq_dim = embedding.shape[1]
        if seq_len > max_len:
            x = embedding[:max_len]
        elif seq_len < max_len:
            x = np.concatenate(
                (embedding, np.zeros((max_len - seq_len, seq_dim))))

        x = torch.from_numpy(x).float()

        return x


def collate_paired_sequences(args):
    x1 = [a[0] for a in args]
    x2 = [a[1] for a in args]
    y = [a[2] for a in args]
    x1 = pad_sequence(x1, batch_first=True)
    x2 = pad_sequence(x2, batch_first=True)
    return x1, x2, torch.stack(y, 0)


class FullyConnectedEmbed(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.nin = nin
        self.nout = nout

        self.transform = nn.Linear(nin, nout)
        self.drop = nn.Dropout(p=0.5)
        self.activation = nn.ReLU()

    def forward(self, x):
        t = self.transform(x)
        t = self.activation(t)
        t = self.drop(t)
        return t


class ContactCNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim=25, ks=7):
        super().__init__()

        self.conv1 = nn.Conv2d(2 * embed_dim, hidden_dim, 1)
        self.batch_norm1 = nn.BatchNorm2d(hidden_dim)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv2d(hidden_dim, 1, ks, padding=ks // 2)
        self.batch_norm2 = nn.BatchNorm2d(1)
        self.activation2 = nn.Sigmoid()
        self.clip()

    def clip(self):
        w = self.conv2.weight
        self.conv2.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, x1, x2):
        x1 = x1.transpose(1, 2)  # b, d, m
        x2 = x2.transpose(1, 2)  # b, d, n

        dif = torch.abs(x1.unsqueeze(3) - x2.unsqueeze(2))  # b, d, m, n
        mul = x1.unsqueeze(3) * x2.unsqueeze(2)
        cat = torch.cat([dif, mul], 1)  # b, 2*d, m, n

        x = self.conv1(cat)
        x = self.activation1(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)  # b, 1, m, n
        x = self.batch_norm2(x)
        x = self.activation2(x)
        x = x.squeeze(1)  # b, m, n
        return x


class LogisticActivation(nn.Module):
    def __init__(self, x0=0.0, k=1, train=False):
        super().__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requires_grad = train

    def forward(self, x):
        o = torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1)
        return o

    def clip(self):
        self.k.data.clamp_(min=0)


class ModelInteraction(nn.Module):
    def __init__(self, embedding, contact):
        super().__init__()
        self.embedding = embedding
        self.contact = contact
        self.activation = LogisticActivation(x0=0.5, k=20)
        self.gamma = nn.Parameter(torch.FloatTensor([0]))
        self.clip()

    def clip(self):
        self.contact.clip()
        self.gamma.data.clamp_(min=0)

    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x = self.contact(x1, x2)  # b, m, n

        mu = torch.mean(x, dim=[1, 2])
        sigma = torch.var(x, dim=[1, 2])
        Q = torch.relu(x - mu - (self.gamma * sigma))
        phat = torch.sum(Q, dim=[1, 2]) / (torch.sum(torch.sign(Q), dim=[1, 2]) + 1)
        phat = self.activation(phat).squeeze()  # b
        return phat


def predict_interaction(model, x1, x2, use_cuda):
    b = x1.shape[0]

    phats = []
    for i in range(b):
        z1 = x1[i:i + 1]
        z2 = x2[i:i + 1]
        if use_cuda:
            z1 = z1.cuda()
            z2 = z2.cuda()

        phat = model(z1, z2)
        phats.append(phat)

    phats = torch.stack(phats)
    return phats


def interaction_grad(model, x1, x2, y, use_cuda=True):
    y_pred = predict_interaction(model, x1, x2, use_cuda)
    if use_cuda:
        y = y.cuda()
    loss = F.binary_cross_entropy(y_pred.float(), y.float())
    loss.backward()
    if use_cuda:
        y_pred = y_pred.cpu()
    return loss.item(), y_pred


def interaction_eval(model, x1, x2, y, use_cuda=True):
    y_pred = predict_interaction(model, x1, x2, use_cuda)
    if use_cuda:
        y = y.cuda()
    loss = F.binary_cross_entropy(y_pred.float(), y.float())
    if use_cuda:
        y_pred = y_pred.cpu()
    return loss.item(), y_pred


def calculate_metrics(labels, phats):
    labels = np.array(labels)
    phats = np.array(phats)
    phats[phats >= 0.5] = 1
    phats[phats < 0.5] = 0
    tn, fp, fn, tp = confusion_matrix(labels, phats).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fpr = fp / (fp + tn)
    return accuracy, recall, precision, fpr
