import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class SkipLSTM(nn.Module):
    def __init__(self, nin, nout, hidden_dim, num_layers, dropout=0, bidirectional=True):
        super().__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        dim = nin
        for i in range(num_layers):
            f = nn.LSTM(
                dim,
                hidden_dim,
                1,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.layers.append(f)
            if bidirectional:
                dim = 2 * hidden_dim
            else:
                dim = hidden_dim

        n = hidden_dim * num_layers + nin
        if bidirectional:
            n = 2 * hidden_dim * num_layers + nin

        self.proj = nn.Linear(n, nout)

    def to_one_hot(self, x):
        """
        Transform numeric encoded amino acid vector to one-hot encoded vector

        :param x: Input numeric amino acid encoding :math:`(N)`
        :type x: torch.Tensor
        :return: One-hot encoding vector :math:`(N \\times n_{in})`
        :rtype: torch.Tensor
        """
        packed = type(x) is PackedSequence
        if packed:
            one_hot = x.data.new(x.data.size(0), self.nin).float().zero_()
            one_hot.scatter_(1, x.data.unsqueeze(1), 1)
            one_hot = PackedSequence(one_hot, x.batch_sizes)
        else:
            one_hot = x.new(x.size(0), x.size(1), self.nin).float().zero_()
            one_hot.scatter_(2, x.unsqueeze(2), 1)
        return one_hot

    def transform(self, x):
        """
        :param x: Input numeric amino acid encoding :math:`(N)`
        :type x: torch.Tensor
        :return: Concatenation of all hidden layers :math:`(N \\times (n_{in} + 2 \\times \\text{num_layers} \\times \\text{hidden_dim}))`
        :rtype: torch.Tensor
        """
        one_hot = self.to_one_hot(x)
        hs = [one_hot]  # []
        h_ = one_hot
        for f in self.layers:
            h, _ = f(h_)
            # h = self.dropout(h)
            hs.append(h)
            h_ = h
        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            h = PackedSequence(h, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
        return h

    def forward(self, x):
        """
        :meta private:
        """
        one_hot = self.to_one_hot(x)
        hs = [one_hot]
        h_ = one_hot

        for f in self.layers:
            h, _ = f(h_)
            # h = self.dropout(h)
            hs.append(h)
            h_ = h

        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
            z = self.proj(h.view(-1, h.size(2)))
            z = z.view(x.size(0), x.size(1), -1)

        return z


class Alphabet:
    """
    From `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.

    :param chars: List of characters in alphabet
    :type chars: byte str
    :param encoding: Mapping of characters to numbers [default: encoding]
    :type encoding: np.ndarray
    :param mask: Set encoding mask [default: False]
    :type mask: bool
    :param missing: Number to use for a value outside the alphabet [default: 255]
    :type missing: int
    """

    def __init__(self, chars, encoding=None, mask=False, missing=255):
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        if encoding is None:
            self.encoding[self.chars] = np.arange(len(self.chars))
            self.size = len(self.chars)
        else:
            self.encoding[self.chars] = encoding
            self.size = encoding.max() + 1
        self.mask = mask
        if mask:
            self.size -= 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x):
        """
        Encode a byte string into alphabet indices

        :param x: Amino acid string
        :type x: byte str
        :return: Numeric encoding
        :rtype: np.ndarray
        """
        x = np.frombuffer(x, dtype=np.uint8)
        return self.encoding[x]

    def decode(self, x):
        """
        Decode numeric encoding to byte string of this alphabet

        :param x: Numeric encoding
        :type x: np.ndarray
        :return: Amino acid string
        :rtype: byte str
        """
        string = self.chars[x]
        return string.tobytes()

    def unpack(self, h, k):
        """unpack integer h into array of this alphabet with length k"""
        n = self.size
        kmer = np.zeros(k, dtype=np.uint8)
        for i in reversed(range(k)):
            c = h % n
            kmer[i] = c
            h = h // n
        return kmer

    def get_kmer(self, h, k):
        """retrieve byte string of length k decoded from integer h"""
        kmer = self.unpack(h, k)
        return self.decode(kmer)


DNA = Alphabet(b"ACGT")


class Uniprot21(Alphabet):
    """
    Uniprot 21 Amino Acid Encoding.

    From `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.
    """

    def __init__(self, mask=False):
        chars = b"ARNDCQEGHILKMFPSTWYVXOUBZ"
        encoding = np.arange(len(chars))
        encoding[21:] = [11, 4, 20, 20]  # encode 'OUBZ' as synonyms
        super().__init__(chars, encoding=encoding, mask=mask, missing=20)


class SDM12(Alphabet):
    """
    A D KER N TSQ YF LIVM C W H G P

    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2732308/#B33
    "Reduced amino acid alphabets exhibit an improved sensitivity and selectivity in fold assignment"
    Peterson et al. 2009. Bioinformatics.
    """

    def __init__(self, mask=False):
        chars = b"ADKNTYLCWHGPXERSQFIVMOUBZ"
        groups = [
            b"A",
            b"D",
            b"KERO",
            b"N",
            b"TSQ",
            b"YF",
            b"LIVM",
            b"CU",
            b"W",
            b"H",
            b"G",
            b"P",
            b"XBZ",
        ]
        groups = {c: i for i in range(len(groups)) for c in groups[i]}
        encoding = np.array([groups[c] for c in chars])
        super().__init__(chars, encoding=encoding, mask=mask)


SecStr8 = Alphabet(b"HBEGITS ")


class seq2tensor(object):
    def __init__(self, filename):
        self.t2v = {}
        self.dim = None
        with open(filename, "r") as fin:
            for line in fin:
                line = line.strip().split("\t")
                t = line[0]
                v = np.array([float(x) for x in line[1].split()])
                if self.dim is None:
                    self.dim = len(v)
                else:
                    v = v[:self.dim]
                self.t2v[t] = v

    def embed(self, seq):
        if seq.find(" ") > 0:
            s = seq.strip().split()
        else:
            s = list(seq.strip())
        rst = []
        for x in s:
            v = self.t2v.get(x)
            if v is None:
                continue
            rst.append(v)
        return np.array(rst)

    def embed_normalized(self, seq, seq_size):
        rst = self.embed(seq)
        if len(rst) > seq_size:
            return rst[:seq_size]
        elif len(rst) < seq_size:
            return np.concatenate(
                (rst, np.zeros((seq_size - len(rst), self.dim))))
        return rst


def read_seq(filename):
    ids = []
    seqs = []
    with open(filename) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ss = line.split("\t")
            if len(ss) != 2:
                continue
            ids.append(ss[0])
            seqs.append(ss[1])
    return ids, seqs