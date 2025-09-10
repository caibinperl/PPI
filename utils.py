import copy
import random

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torch.utils.data import Dataset, DataLoader

plt.style.use('fivethirtyeight')


def plot_losses(losses, val_losses):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(losses, label='Training Loss', c='b')
    plt.plot(val_losses, label='Validation Loss', c='r')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    return fig


def calculate_metrics(y_true, y_pred):
    y_pred = (y_pred >= 0.5).astype(np.int32)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    return acc, precision, recall, f1


def pot_metrics(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)

    ax2.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    positive_ratio = np.sum(y_true) / len(y_true)
    ax2.axhline(y=positive_ratio, color='r', linestyle='--',
                label='Random classifier')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="upper right")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")


def train_model(model, train_loader, val_loader, optimizer, epochs, device):
    losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for x1, x2, y in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            yhat = model(x1, x2)
            batch_loss = F.binary_cross_entropy(yhat, y)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_losses.append(batch_loss.item())

        loss = np.mean(batch_losses)
        losses.append(loss)

        model.eval()
        with torch.no_grad():
            batch_losses = []
            n = 0
            accuracy_average = 0
            for x1, x2, y in val_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)
                yhat = model(x1, x2)
                batch_loss = F.binary_cross_entropy(yhat, y)
                batch_losses.append(batch_loss.item())
                accuracy = accuracy_score(y.cpu(), (yhat.cpu() > 0.5).int())
                accuracy_average = (accuracy_average * n + accuracy * x1.shape[
                    0]) / (n + x1.shape[0])
                n = n + x1.shape[0]

            val_loss = np.mean(batch_losses)
            val_losses.append(val_loss)

        print(
            f"Epoch: {epoch + 1} -- loss: {loss:.4f}, val_loss: {val_loss:.4f}, accuracy: {accuracy_average:.4f}")

    plot_losses(losses, val_losses)


def eval_model(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []

        for x1, x2, y in val_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            yhat = model(x1, x2)
            y_true.append(y.cpu().numpy())
            y_pred.append(yhat.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        acc, precision, recall, f1 = calculate_metrics(y_true, y_pred)
        print(
            f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        pot_metrics(y_true, y_pred)


def save_model(model, model_path):
    model.to("cpu")
    torch.save(model.state_dict(), model_path)


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

    def loader(self, id, max_len=1500):
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


def load_data(data_file, batch_size, embedding_h5, train=True):
    df = pd.read_csv(data_file, sep="\t", header=None)
    dataset = PairedDataset(df[0].to_list(), df[1].to_list(),
                            df[2].to_list(), embedding_h5)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return loader


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Transformer

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        angular_speed = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * angular_speed)  # even dimensions
        pe[:, 1::2] = torch.cos(position * angular_speed)  # odd dimensions
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x is N, L, D
        # pe is 1, maxlen, D
        scaled_x = x * np.sqrt(self.d_model)
        encoded = scaled_x + self.pe[:, :x.size(1), :]
        return encoded


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = int(d_model / n_heads)
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.alphas = None

    def make_chunks(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # N, L, D -> N, L, n_heads * d_k
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        # N, n_heads, L, d_k
        x = x.transpose(1, 2)
        return x

    def init_keys(self, key):
        # N, n_heads, L, d_k
        self.proj_key = self.make_chunks(self.linear_key(key))
        self.proj_value = self.make_chunks(self.linear_value(key))

    def score_function(self, query):
        # scaled dot product
        # N, n_heads, L, d_k x # N, n_heads, d_k, L -> N, n_heads, L, L
        proj_query = self.make_chunks(self.linear_query(query))
        dot_products = torch.matmul(proj_query,
                                    self.proj_key.transpose(-2, -1))
        scores = dot_products / np.sqrt(self.d_k)
        return scores

    def attn(self, query, mask=None):
        # Query is batch-first: N, L, D
        # Score function will generate scores for each head
        scores = self.score_function(query)  # N, n_heads, L, L
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        alphas = F.softmax(scores, dim=-1)  # N, n_heads, L, L
        alphas = self.dropout(alphas)
        self.alphas = alphas.detach()

        # N, n_heads, L, L x N, n_heads, L, d_k -> N, n_heads, L, d_k
        context = torch.matmul(alphas, self.proj_value)
        return context

    def output_function(self, contexts):
        # N, L, D
        out = self.linear_out(contexts)  # N, L, D
        return out

    def forward(self, query, mask=None):
        if mask is not None:
            # N, 1, L, L - every head uses the same mask
            mask = mask.unsqueeze(1)

        # N, n_heads, L, d_k
        context = self.attn(query, mask=mask)
        # N, L, n_heads, d_k
        context = context.transpose(1, 2).contiguous()
        # N, L, n_heads * d_k = N, L, d_model
        context = context.view(query.size(0), -1, self.d_model)
        # N, L, d_model
        out = self.output_function(context)
        return out


class SubLayerWrapper(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, sublayer, is_self_attn=False, **kwargs):
        norm_x = self.norm(x)
        if is_self_attn:
            sublayer.init_keys(norm_x)
        out = x + self.drop(sublayer(norm_x, **kwargs))
        return out


class TransformerLayer(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, dropout=0.2):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_units = ff_units
        self.attn_heads = MultiHeadedAttention(n_heads, d_model,
                                               dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model),
        )
        self.sublayers = nn.ModuleList(
            [SubLayerWrapper(d_model, dropout) for _ in range(2)])

    def forward(self, query, mask=None):
        # SubLayer 0 - Self-Attention
        att = self.sublayers[0](query,
                                sublayer=self.attn_heads,
                                is_self_attn=True,
                                mask=mask)
        # SubLayer 1 - FFN
        out = self.sublayers[1](att, sublayer=self.ffn)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, layer, n_layers=1, max_len=10000):
        super().__init__()

        self.d_model = layer.d_model
        self.pe = PositionalEncoding(max_len, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.layers = nn.ModuleList([copy.deepcopy(layer)
                                     for _ in range(n_layers)])

    def forward(self, query, mask=None):
        # Positional Encoding
        x = self.pe(query)
        for layer in self.layers:
            x = layer(x, mask)
        # Norm
        return self.norm(x)
