import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

plt.style.use('fivethirtyeight')


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

    def loader(self, id, max_len=2000):
        x = self.embed_data[id]
        seq_len = x.shape[0]
        seq_dim = x.shape[1]
        if seq_len > max_len:
            x = x[:max_len]
        elif seq_len < max_len:
            x = np.concatenate(
                (x, np.zeros((max_len - seq_len, seq_dim))))

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

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True,
                           batch_first=True)
        self.conv2 = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3,
                               padding=1)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True,
                           batch_first=True)
        self.conv3 = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3,
                               padding=1)
        self.gru3 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True,
                           batch_first=True)
        self.conv4 = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3,
                               padding=1)
        self.gru4 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True,
                           batch_first=True)
        self.conv5 = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3,
                               padding=1)
        self.gru5 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True,
                           batch_first=True)
        self.conv6 = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3,
                               padding=1)

        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.leaky_relu = nn.LeakyReLU(0.3)

        self.fc1 = nn.Linear(hidden_dim, 100)
        self.fc2 = nn.Linear(100, (hidden_dim + 7) // 2)
        self.fc3 = nn.Linear((hidden_dim + 7) // 2, 1)

        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d((hidden_dim + 7) // 2)

    def process_sequence(self, x):
        # x (N, F=13, L=2000)
        # 第一层
        x = self.conv1(x)  # N, F=10, L=2000,
        x = self.pool(x)  # N, F=10, L= 666,
        x = x.permute(0, 2, 1)  # N, L=666, F=10
        gru_out, _ = self.gru1(x)  # N, L=666, F=20
        x = x.permute(0, 2, 1)  # N, F=10, L=666
        gru_out = gru_out.permute(0, 2, 1)  # N, F=20, L=666
        x = torch.cat([gru_out, x], dim=1)  # [8, 30, 666]

        # 第二层
        x = self.conv2(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru2(x)
        x = x.permute(0, 2, 1)
        gru_out = gru_out.permute(0, 2, 1)
        x = torch.cat([gru_out, x], dim=1)  # [8, 30, 222]

        # 第三层
        x = self.conv3(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru3(x)
        x = x.permute(0, 2, 1)
        gru_out = gru_out.permute(0, 2, 1)
        x = torch.cat([gru_out, x], dim=1)  # [8, 30, 74]

        # 第四层
        x = self.conv4(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru4(x)
        x = x.permute(0, 2, 1)
        gru_out = gru_out.permute(0, 2, 1)
        x = torch.cat([gru_out, x], dim=1)  # [8, 30, 24]

        # 第五层
        x = self.conv5(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru5(x)
        x = x.permute(0, 2, 1)
        gru_out = gru_out.permute(0, 2, 1)
        x = torch.cat([gru_out, x], dim=1)  # [8, 30, 8]

        # 第六层
        x = self.conv6(x)  # [8, 10, 8]

        x = self.adaptive_pool(x).squeeze(-1)  # [8, 10]
        return x

    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 1)  # N, F, L
        x1 = self.process_sequence(x1)

        x2 = x2.permute(0, 2, 1)
        x2 = self.process_sequence(x2)

        merged = x1 * x2

        x = self.fc1(merged)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = torch.flatten(x)
        x = F.sigmoid(x)
        return x


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
