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

# ============ embedding ===============

class FullyConnectedEmbed(nn.Module):
    """
    Protein Projection Module. Takes embedding from language model and outputs 
    low-dimensional interaction aware projection.

    Args:
        nin (int): Size of language model output
        nout (int): Dimension of projection
        dropout (float): Proportion of weights to drop out [default: 0.5]
        activation (nn.Module): Activation for linear projection model
    """

    def __init__(self, nin, nout, dropout=0.5):
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.dropout_p = dropout

        self.transform = nn.Linear(nin, nout)
        self.drop = nn.Dropout(p=self.dropout_p)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input language model embedding :math:`(b \\times N \\times d_0)`

        Returns:
            torch.Tensor: Low dimensional projection of embedding
        """
        t = self.transform(x)
        t = self.activation(t)
        t = self.drop(t)
        return t


# ========== contact ============

class FullyConnected(nn.Module):
    """
    Performs part 1 of Contact Prediction Module. Takes embeddings from 
    Projection module and produces broadcast tensor.

    Input embeddings of dimension :math:`d` are combined into a :math:`2d` 
    length MLP input :math:`z_{cat}`, where :math:`z_{cat} = [z_0 \\ominus z_1 | z_0 \\odot z_1]`

    Args:
        embed_dim (int): Output dimension of `dscript.model.embedding` model :math:`d` [default: 100]
        hidden_dim (int): Hidden dimension :math:`h` [default: 50]
    """

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()

        self.D = embed_dim
        self.H = hidden_dim
        self.conv = nn.Conv2d(2 * self.D, self.H, 1)
        self.batchnorm = nn.BatchNorm2d(self.H)
        self.activation = nn.ReLU()

    def forward(self, x0, x1):
        """
        Args:
            x0 (torch.Tensor): Projection module embedding :math:`(b \\times m \\times d)`
            x1 (torch.Tensor): Projection module embedding :math:`(b \\times n \\times d)`

        Returns:
            torch.Tensor: Predicted broadcast tensor :math:`(b \\times m \\times n \\times h)`
        """
        x0 = x0.transpose(1, 2)  # (b, d, m)
        x1 = x1.transpose(1, 2)  # (b, d, n)

        z_dif = torch.abs(x0.unsqueeze(3) - x1.unsqueeze(2))  # (b, d, n)
        z_mul = x0.unsqueeze(3) * x1.unsqueeze(2)
        z_cat = torch.cat([z_dif, z_mul], 1)

        B = self.conv(z_cat)
        B = self.activation(B)
        B = self.batchnorm(B)

        return B


class ContactCNN(nn.Module):
    """
    Residue Contact Prediction Module. Takes embeddings from Projection module 
    and produces contact map, output of Contact module.

    Args:
        embed_dim (int): Output dimension of `dscript.model.embedding` (d) 
        hidden_dim (int): Hidden dimension (h)
        width (int): Width of convolutional filter (2w+)
    """

    def __init__(self, embed_dim, hidden_dim=50, width=7):
        super().__init__()
        self.fully_connect = FullyConnected(embed_dim, hidden_dim)
        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.batchnorm = nn.BatchNorm2d(1)
        self.activation = nn.Sigmoid()
        self.clip()

    def clip(self):
        """
        Force the convolutional layer to be transpose invariant.
        """
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, x0, x1):
        """
        Args:
            x0 (torch.Tensor): Projection module embedding :math:`(b \\times n \\times d)`
            x1 (torch.Tensor): Projection module embedding :math:`(b \\times m \\times d)`

        Returns:
            torch.Tensor: Predicted contact map :math:`(b \\times n \\times m)`
        """
        B = self.fully_connect(x0, x1)  # (b, h, n, m)
        C = self.conv(B)  # (b, 1, n, m)
        C = self.batchnorm(C)
        C = self.activation(C)
        return C


# ============ interaction ===============

class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:

    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`

    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super().__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requires_grad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise

        :param x: :math:`(N \\times *)` where :math:`*` means, any number of additional dimensions
        :type x: torch.Tensor
        :return: :math:`(N \\times *)`, same shape as the input
        :rtype: torch.Tensor
        """
        o = torch.clamp(
            1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1)
        return o

    def clip(self):
        """
        Restricts sigmoid slope :math:`k` to be greater than or equal to 0, if :math:`k` is trained.

        :meta private:
        """
        self.k.data.clamp_(min=0)


class ModelInteraction(nn.Module):
    def __init__(
        self,
        embedding,
        contact,
        do_w=True,
        do_sigmoid=True,
        do_pool=False,
        pool_size=9,
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
    ):
        """
        Main D-SCRIPT model. Contains an embedding and contact model and offers 
        access to those models. Computes pooling operations on contact map to 
        generate interaction probability.

        Args:
            embedding (dscript.model.embedding.FullyConnectedEmbed): Embedding model
            contact (dscript.model.contact.ContactCNN): Contact model
            do_w (bool): whether to use the weighting matrix [default: True]
            do_sigmoid (bool): whether to use a final sigmoid activation [default: True]
            do_pool (bool): whether to do a local max-pool prior to the global pool
            pool_size (int): width of max-pool [default 9]
            theta_init (float): initialization value of :math:`\\theta` for weight matrix [default: 1]
            lambda_init (float): initialization value of :math:`\\lambda` for weight matrix [default: 0]
            gamma_init (float): initialization value of :math:`\\gamma` for global pooling [default: 0]
        """
        super().__init__()
        self.do_w = do_w
        self.do_sigmoid = do_sigmoid
        if do_sigmoid:
            self.activation = LogisticActivation(x0=0.5, k=20)

        self.embedding = embedding
        self.contact = contact

        if self.do_w:
            self.theta = nn.Parameter(torch.FloatTensor([theta_init]))
            self.lambda_ = nn.Parameter(torch.FloatTensor([lambda_init]))

        self.do_pool = do_pool
        self.pool_size = pool_size
        self.max_pool = nn.MaxPool2d(pool_size, padding=pool_size // 2)

        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))

        self.clip()

        self.xx = nn.Parameter(torch.arange(2000), requires_grad=False)

    def clip(self):
        """
        Clamp model values
        """
        self.contact.clip()

        if self.do_w:
            self.theta.data.clamp_(min=0, max=1)
            self.lambda_.data.clamp_(min=0)

        self.gamma.data.clamp_(min=0)

    def map_predict(self, x0, x1):
        """
        Project down input language model embeddings into low dimension using projection module

        Args:
            x0 (torch.Tensor): Language model embedding :math:`(b \\times m \\times d_0)`
            x1 (torch.Tensor): Language model embedding :math:`(b \\times n \\times d_0)`
        Returns:
            torch.Tensor, torch.Tensor: Predicted contact map, predicted probability of interaction :math:`(b \\times N \\times d_0), (1)`
        """
        e0 = self.embedding(x0)  # (b, m, d)
        e1 = self.embedding(x1)  # (b, n, d)
        C = self.contact(e0, e1)

        if self.do_w:
            N, M = C.shape[2:]

            a = -1 * torch.square(
                (self.xx[:N] + 1 - ((N + 1) / 2)) / (-1 * ((N + 1) / 2))
            )

            b = -1 * torch.square(
                (self.xx[:M] + 1 - ((M + 1) / 2)) / (-1 * ((M + 1) / 2))
            )

            a = torch.exp(self.lambda_ * a)
            b = torch.exp(self.lambda_ * b)

            W = a.unsqueeze(1) * b
            W = (1 - self.theta) * W + self.theta
            yhat = C * W
        else:
            yhat = C

        if self.do_pool:
            yhat = self.max_pool(yhat)

        # Mean of contact predictions where p_ij > mu + gamma*sigma
        mu = torch.mean(yhat, dim=(1, 2, 3)).repeat(
            yhat.shape[2]*yhat.shape[3], 1).T.reshape(yhat.shape[0], yhat.shape[1], yhat.shape[2], yhat.shape[3])
        sigma = torch.var(yhat, dim=(1, 2, 3)).repeat(
            yhat.shape[2]*yhat.shape[3], 1).T.reshape(yhat.shape[0], yhat.shape[1], yhat.shape[2], yhat.shape[3])
        Q = torch.relu(yhat - mu - (self.gamma * sigma))
        phat = torch.sum(Q, dim=(1, 2, 3)) / \
            (torch.sum(torch.sign(Q), dim=(1, 2, 3)) + 1)
        if self.do_sigmoid:
            phat = self.activation(phat)
        return C, phat

    def forward(self, x0, x1):
        _, phat = self.map_predict(x0, x1)
        return phat

# Trainer


class Trainer(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        self.losses = []
        self.val_losses = []

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
