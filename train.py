#!/usr/bin/env python3
# train.py
"""Train a uPIT BLSTM

Usage:
    train.py [options]

Options:
    -d, --debug                     # output extra debug information
    --debug-level LEVEL             # set the debug level to LEVEL [default: 1]
    --epochs EPOCHS                 # set the number of epochs [default: 10]
    --batch-size BATCH-SIZE         # set batch size [default: 8]
    --num-layers NUM-LAYERS         # number of hidden layers [default: 3]
    --layer-size LAYER-SIZE         # size of each hidden layer [default: 896]
    --dropout DROPOUT               # dropout rate [default: 0.5]
    --hop-size HOP-SIZE             # hop size [default: 128]
    --window-size WINDOW-SIZE       # window size [default: 256]
    --window WINDOW                 # window type [default: Hann]
    --learning-rate LEARNING-RATE   # learning rate [default: 1.0e-3]
    --clip-norm CLIP-NORM           # clip norm [default: 200]
    --weight-decay WEIGHT-DECAY     # weight decay [default: 1.0e-5]

Examples:
    ./train.py -d --epochs 2
"""

from docopt import docopt

import torch as th
import numpy as np

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

class UPITbLSTM(th.nn.Module):
    def __init__(self, num_bins, num_layers=3, layer_size=896, dropout=0.5):
        super(UPITbLSTM, self).__init__()
        self.num_s = 2
        self.blstm = th.nn.LSTM(num_bins,
                layer_size,
                num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True)
        self.drop = th.nn.Dropout(p=dropout)
        self.linear = th.nn.ModuleList([
            th.nn.Linear(layer_size * 2, num_bins)
            for _ in range(self.num_s)])
        self.nonlinear = th.nn.functional.relu
        self.num_bins = num_bins

    def forward(self, x, train=True):
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

class TrainUpit(object):
    def __init__(self, args):
        self.hop_size = int(args['--hop-size'])
        self.window_size = int(args['--window-size'])
        self.window = args['--window']
        self.num_bins = fft_size(self.window_size) // 2 + 1

        print(f"Window size: {self.window_size}\n"
                f"Hop size: {self.hop_size}\n"
                f"Window: {self.window}")

        self.epochs = int(args['--epochs'])
        self.num_layers = int(args['--num-layers'])
        self.layer_size = int(args['--layer-size'])
        self.batch_size = int(args['--batch-size'])
        self.dropout = float(args['--dropout'])

        print(f"Epochs: {self.epochs}\n"
                f"Hidden layers: {self.num_layers}\n"
                f"Hidden layer size: {self.layer_size}\n"
                f"Dropout: {self.dropout}")

        self.lr = float(args['--learning-rate'])
        # self.clip_norm  = float(args['--clip-norm'])
        self.weight_decay  = float(args['--weight-decay'])

        print(f"Learning rate: {self.lr}\nWeight Decay: {self.weight_decay}")

        self.model = UPITbLSTM(self.num_bins,
                self.num_layers, self.layer_size, self.dropout)
        print(self.model)

        # load training data
        # uttloader creates a Dataset which is passed to a DataLoader and returned
        # DataLoader implements __iter__(self) and yields the data in batches
        # load cv data

    def train(self):
        self.optimizer = th.optim.Adam(self.model.parameters(),
                lr=self.lr, weight_decay=self.weight_decay)
        pass


def main(args):
    global DEBUG, DEBUG_LEVEL
    print(arguments)

    DEBUG = args['--debug']
    DEBUG_LEVEL = int(args['--debug-level'])

    trainer = TrainUpit(args)

    trainer.train()


defaults = {
    '--debug': False,
}

if __name__ == "__main__":
    arguments = docopt(__doc__)

    # Drop arguments that are None
    arguments = {key:val for key, val in arguments.items() if val is not None}
    # merge dictionaries
    arguments = {**defaults, **arguments}

    main(arguments)
