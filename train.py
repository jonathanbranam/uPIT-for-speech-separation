#!/usr/bin/env python3
# train.py
"""Train a uPIT BLSTM

Usage:
    train.py [--epochs EPOCHS]
        [-d] [--debug-level LEVEL]

Options:
    -d, --debug                       # output extra debug information
    --debug-level LEVEL               # set the debug level to LEVEL

Examples:
    ./train.py -d --epochs 2
"""

from docopt import docopt

import torch as th
import numpy as np

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

class UPITbLSTM(th.nn.Module):
    def __init__(self, num_bins, num_layers=3, hidden_size=896, dropout=0.5):
        super(UPITbLSTM, self).__init__()
        self.num_s = 2
        self.blstm = th.nn.LSTM(num_bins,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True)
        self.drop = th.nn.Dropout(p=dropout)
        self.linear = th.nn.ModuleList([
            th.nn.Linear(hidden_size * 2, num_bins)
            for _ in range(self.num_s)])
        self.nonlinear = th.nn.functional.relu
        self.num_bins = num_bins

    def feed_forward(self, x, train=True):
        is_packed = isinstance(x, PackedSequence)
        # extend dim when inference
        if not is_packed and x.dim() != 3:
            x = th.unsqueeze(x, 0)
        x, _ = self.blstm(x)
        # using unpacked sequence
        # x: N x T x D
        if is_packed:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.drop(x)
        lin_list = []
        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            if not train:
                y = y.view(-1, self.num_bins)
            lin_list.append(y)
        return lin_list


def fft_size(window_size):
    return int(2**np.ceil(int(np.log2(window_size))))


def main(args):
    print(arguments)
    print("main()")

    hop_size = 128
    window_size = 256
    window = "Hann"
    num_bins = fft_size(window_size) // 2 + 1

    model = UPITbLSTM(num_bins)

    # load training data
    # load cv data


defaults = {
    '--debug': False,
    '--debug-level': 1,
    'EPOCHS': 10,
}

if __name__ == "__main__":
    arguments = docopt(__doc__)

    # Drop arguments that are None
    arguments = {key:val for key, val in arguments.items() if val is not None}
    # merge dictionaries
    arguments = {**defaults, **arguments}

    main(arguments)
