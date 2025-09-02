import copy

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

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


# Data loader

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


# Model

class Featuring(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()

        self.feature_dim = feature_dim

        self.conv1 = nn.Conv1d(input_dim, feature_dim, kernel_size=3,
                               padding=1)
        layer1 = TransformerLayer(n_heads=3, d_model=feature_dim,
                                  ff_units=10, dropout=0.2)
        self.encoder1 = TransformerEncoder(layer1, n_layers=2)

        self.conv2 = nn.Conv1d(2 * feature_dim, feature_dim, kernel_size=3,
                               padding=1)
        layer2 = TransformerLayer(n_heads=3, d_model=feature_dim,
                                  ff_units=10, dropout=0.2)
        self.encoder2 = TransformerEncoder(layer2, n_layers=2)

        self.conv3 = nn.Conv1d(2 * feature_dim, feature_dim, kernel_size=3,
                               padding=1)
        layer3 = TransformerLayer(n_heads=3, d_model=feature_dim,
                                  ff_units=10, dropout=0.2)
        self.encoder3 = TransformerEncoder(layer3, n_layers=2)

        self.conv4 = nn.Conv1d(2 * feature_dim, feature_dim, kernel_size=3,
                               padding=1)
        layer4 = TransformerLayer(n_heads=3, d_model=feature_dim,
                                  ff_units=10, dropout=0.2)
        self.encoder4 = TransformerEncoder(layer4, n_layers=2)

        self.conv5 = nn.Conv1d(2 * feature_dim, feature_dim, kernel_size=3,
                               padding=1)

        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # b, input_dim, 1800

        # First layer
        x = self.conv1(x)  # b, feature_dim, 1800,
        x = self.pool(x)  # b, feature_dim, 600
        x = x.permute(0, 2, 1)  # n, 600, feature_dim
        e = self.encoder1(x)  # b, 600, feature_dim
        x = x.permute(0, 2, 1)  # b, feature_dim, 600
        e = e.permute(0, 2, 1)  # b, feature_dim, 600
        x = torch.cat([e, x], dim=1)  # b, 2*feature_dim, 600

        # Second layer
        x = self.conv2(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        e = self.encoder2(x)
        x = x.permute(0, 2, 1)
        e = e.permute(0, 2, 1)
        x = torch.cat([e, x], dim=1)  # b, 2*feature_dim, 200

        # Third layer
        x = self.conv3(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        e = self.encoder3(x)
        x = x.permute(0, 2, 1)
        e = e.permute(0, 2, 1)
        x = torch.cat([e, x], dim=1)  # b, 2*feature_dim, 66

        # Fourth layer
        x = self.conv4(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        e = self.encoder4(x)
        x = x.permute(0, 2, 1)
        e = e.permute(0, 2, 1)
        x = torch.cat([e, x], dim=1)  # b, 2*feature_dim, 22

        x = self.conv5(x)  # b, feature_dim, 22
        # x = self.adaptive_pool(x) # b, feature_dim, 1
        # x = x.squeeze(-1)  # b, feature_dim
        x = x.permute(0, 2, 1)  # b, 22, feature_dim
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc5(x)

        x = x.squeeze(-1)
        x = F.sigmoid(x)

        return x


class InteractionModel(nn.Module):
    def __init__(self, featuring, classifier):
        super().__init__()

        self.featuring = featuring
        self.classifier = classifier
        layer = TransformerLayer(n_heads=3, d_model=2 * self.featuring.feature_dim,
                                 ff_units=10, dropout=0.5)
        self.encoder = TransformerEncoder(layer, n_layers=2)

    def forward(self, x1, x2):
        x1 = self.featuring(x1)  # b, 22, feature_dim
        x2 = self.featuring(x2)  # b, 22, feature_dim
        x = torch.cat((x1, x2), dim=2)  # b, 22, 2*feature_dim
        x = self.encoder(x)
        x = torch.mean(x, dim=1) # b, 2*feature_dim
        x = self.classifier(x)
        return x


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
