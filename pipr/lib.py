import torch
import torch.nn as nn
import h5py
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use('fivethirtyeight')


# Data loader

class PairedDataset(Dataset):

    def __init__(self, x0, x1, y, embedding_h5):
        self.x0 = x0
        self.x1 = x1
        self.y = y
        self.embed_data = {}
        
        ids = set(x0).union(set(x1))
        with h5py.File(embedding_h5, "r") as h5fin:
            for id in ids:
                self.embed_data[id] = h5fin[id][:, :]

    def __len__(self):
        return len(self.x0)
    
    def __getitem__(self, i):
        z0 = torch.from_numpy(self.embed_data[self.x0[i]])
        z1 = torch.from_numpy(self.embed_data[self.x1[i]])
        return z0, z1, torch.as_tensor(self.y[i]).float()


def collate_paired_sequences(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    x0 = pad_sequence(x0, batch_first=True)
    x1 = pad_sequence(x1, batch_first=True)
    return x0, x1, torch.stack(y, 0)


# Model

class ProteinInteractionModel(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(ProteinInteractionModel, self).__init__()
        
        # 共享层（两个输入分支使用相同的层）
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.conv2 = nn.Conv1d(3*hidden_dim, hidden_dim, kernel_size=3, padding=1)  # 3*hidden_dim因为拼接了GRU输出
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.conv3 = nn.Conv1d(3*hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gru3 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.conv4 = nn.Conv1d(3*hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gru4 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.conv5 = nn.Conv1d(3*hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gru5 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.conv6 = nn.Conv1d(3*hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.leaky_relu = nn.LeakyReLU(0.3)
        
        # 分类器
        self.fc1 = nn.Linear(hidden_dim, 100)
        self.fc2 = nn.Linear(100, (hidden_dim + 7) // 2)
        self.fc3 = nn.Linear((hidden_dim + 7) // 2, 1)
        
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d((hidden_dim + 7) // 2)

    def forward(self, x1, x2):
        # 处理第一个序列
        x1 = x1.permute(0, 2, 1)  # N, F, L
        x1 = self.process_sequence(x1)
        # print(f"x1: {x1.shape}")
        
        # 处理第二个序列
        x2 = x2.permute(0, 2, 1)
        x2 = self.process_sequence(x2)
        # print(f"x2: {x2.shape}")
        
        # 合并特征
        merged = x1 * x2
        # print(f"merged: {merged.shape}")
        
        # 分类器
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
        return x
    
    def process_sequence(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # GRU需要序列长度在第二维
        gru_out, _ = self.gru1(x)
        x = x.permute(0, 2, 1)  # 恢复通道优先, N, F, L
        gru_out = gru_out.permute(0, 2, 1)
        x = torch.cat([gru_out, x], dim=1)
        
        # 第二层
        x = self.conv2(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru2(x)
        x = x.permute(0, 2, 1)
        gru_out = gru_out.permute(0, 2, 1)
        x = torch.cat([gru_out, x], dim=1)
        
        # 第三层
        x = self.conv3(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru3(x)
        x = x.permute(0, 2, 1)
        gru_out = gru_out.permute(0, 2, 1)
        x = torch.cat([gru_out, x], dim=1)
        
        # 第四层
        x = self.conv4(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru4(x)
        x = x.permute(0, 2, 1)
        gru_out = gru_out.permute(0, 2, 1)
        x = torch.cat([gru_out, x], dim=1)
        
        # 第五层
        x = self.conv5(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru5(x)
        x = x.permute(0, 2, 1)
        gru_out = gru_out.permute(0, 2, 1)
        x = torch.cat([gru_out, x], dim=1)
        
        # 第六层
        x = self.conv6(x)
        x = self.adaptive_pool(x).squeeze(-1)
        return x
    

# Train

class Trainer(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(
                f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _make_train_step_fn(self):
        def perform_train_step_fn(x0, x1, y):
            self.model.train()
            yhat = self.model(x0, x1)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()

        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(x0, x1, y):
            self.model.eval()
            yhat = self.model(x0, x1)
            loss = self.loss_fn(yhat, y)
            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        n_batches = len(data_loader)
        mini_batch_losses = []
        for i, (x0_batch, x1_batch, y_batch) in enumerate(data_loader):
            x0_batch = x0_batch.to(self.device)
            x1_batch = x1_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x0_batch, x1_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=1234):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def train(self, n_epochs, seed=1234):
        self.set_seed(seed)

        for epoch in tqdm(range(n_epochs)):
            self.total_epochs += 1

            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

    def predict(self, x):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer