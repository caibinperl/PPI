import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
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

    def loader(self, id, max_len=1800):
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
    def __init__(self, embed_dim):
        super().__init__()

        hidden_dim = 50
        width = 7

        self.conv1 = nn.Conv2d(2 * embed_dim, hidden_dim, 1)
        self.batch_norm1 = nn.BatchNorm2d(hidden_dim)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.batch_norm2 = nn.BatchNorm2d(1)
        self.activation2 = nn.Sigmoid()
        self.clip()

    def clip(self):
        w = self.conv2.weight
        self.conv2.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, x1, x2):
        x1 = x1.transpose(1, 2)  # b, embed_dim, m
        x2 = x2.transpose(1, 2)  # b, embed_dim, n

        dif = torch.abs(x1.unsqueeze(3) - x2.unsqueeze(2))  # b, embed_dim, m, n
        mul = x1.unsqueeze(3) * x2.unsqueeze(2)
        cat = torch.cat([dif, mul], 1)  # b, 2*embed_dim, m, n

        x = self.conv1(cat)
        x = self.activation1(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)  # b, 1, m, n
        x = self.batch_norm2(x)
        x = self.activation2(x)
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
    def __init__(self, embedding, contact, use_cuda):
        super().__init__()
        gamma_init = 0
        self.embedding = embedding
        self.contact = contact
        self.use_cuda = use_cuda
        self.activation = LogisticActivation(x0=0.5, k=20)
        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))
        self.clip()

    def clip(self):
        self.contact.clip()
        self.gamma.data.clamp_(min=0)

    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        C = self.contact(x1, x2)  # b, 1, m, n
        yhat = C.unsqueeze(1)  # b, m, n

        mu = torch.mean(yhat, dim=[1, 2]).view(-1, 1, 1)
        sigma = torch.var(yhat, dim=[1, 2]).view(-1, 1, 1)
        Q = torch.relu(yhat - mu - (self.gamma * sigma))
        phat = torch.sum(Q, dim=[1, 2]) / (torch.sum(torch.sign(Q), dim=[1, 2]) + 1)
        # phat (b,)
        phat = self.activation(phat)
        c_map = torch.mean(yhat, dim=[1, 2])
        return c_map, phat


def predict_cmap_interaction(model, x1, x2, use_cuda):
    if use_cuda:
        x1 = x1.cuda()
        x2 = x2.cuda()

    c_map, p_hat = model(x1, x2)
    return c_map, p_hat


def interaction_grad(model, x1, x2, y, use_cuda=True):
    accuracy_weight = 0.35

    c_map, p_hat = predict_cmap_interaction(model, x1, x2, use_cuda)

    if use_cuda:
        y = y.cuda()
    y = Variable(y)

    p_hat = p_hat.float()
    bce_loss = F.binary_cross_entropy(p_hat.float(), y.float())
    accuracy_loss = bce_loss
    representation_loss = c_map
    loss = (accuracy_weight * accuracy_loss) + ((1 - accuracy_weight) * representation_loss)

    # Backprop Loss
    loss.backward()

    if use_cuda:
        y = y.cpu()
        p_hat = p_hat.cpu()

    b = p_hat.shape[0]
    with torch.no_grad():
        guess_cutoff = 0.5
        p_hat = p_hat.float()
        p_guess = (guess_cutoff * torch.ones(b) < p_hat).float()
        y = y.float()
        correct = torch.sum(p_guess == y).item()
        mse = torch.mean((y.float() - p_hat) ** 2).item()

    return loss, correct, mse, b


def predict_interaction(model, n0, n1, use_cuda):
    _, p_hat = predict_cmap_interaction(model, n0, n1, use_cuda)
    return p_hat


def interaction_eval(model, val_loader, use_cuda):
    p_hat = []
    true_y = []

    for n0, n1, y in val_loader:
        ph = predict_interaction(model, n0, n1, use_cuda)
        p_hat.append(ph)
        true_y.append(y)

    y = torch.cat(true_y, 0)
    p_hat = torch.cat(p_hat, 0)

    if use_cuda:
        y.cuda()
        p_hat = torch.Tensor([x.cuda() for x in p_hat])
        p_hat.cuda()

    loss = F.binary_cross_entropy(p_hat.float(), y.float()).item()
    b = y.shape[0]

    with torch.no_grad():
        guess_cutoff = torch.Tensor([0.5]).float()
        p_hat = p_hat.float()
        y = y.float()
        p_guess = (guess_cutoff * torch.ones(b) < p_hat).float()
        correct = torch.sum(p_guess == y).item()
        mse = torch.mean((y.float() - p_hat) ** 2).item()

        tp = torch.sum(y * p_hat).item()
        pr = tp / torch.sum(p_hat).item()
        re = tp / torch.sum(y).item()
        f1 = 2 * pr * re / (pr + re)

    y = y.cpu().numpy()
    p_hat = p_hat.data.cpu().numpy()

    aupr = average_precision_score(y, p_hat)

    return loss, correct, mse, pr, re, f1, aupr
