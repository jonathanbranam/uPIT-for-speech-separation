#!/usr/bin/env python3
# train.py
"""Train a uPIT BLSTM

Usage:
    train.py [options]

Options:
    -d, --debug                     # output extra debug information
    --debug-level LEVEL             # set the debug level to LEVEL [default: 1]
                                    #
    --train-root-dir TRAIN-ROOT-DIR # training set root directory [default: data/2spk8kmax/tr]
    --valid-root-dir VALID-ROOT-DIR # validation set root directory [default: data/2spk8kmax/cv]
    --mix-dir MIX-DIR               # directory beneath root with mixtures [default: mix]
    --src1-dir SRC1-DIR             # directory beneath root with source 1 [default: s1]
    --src2-dir SRC2-DIR             # directory beneath root with source 2 [default: s2]
    --train-file-list TRAIN-FL      # file that contains training set [default: train.txt]
    --valid-file-list VAL-FL        # file that contains validation set [default: validation.txt]
                                    #
    --epochs EPOCHS                 # set the number of epochs [default: 10]
    --batch-size BATCH-SIZE         # set batch size [default: 8]
    --num-layers NUM-LAYERS         # number of hidden layers [default: 3]
    --layer-size LAYER-SIZE         # size of each hidden layer [default: 896]
    --dropout DROPOUT               # dropout rate [default: 0.5]
                                    #
    --hop-size HOP-SIZE             # hop size [default: 128]
    --window-size WINDOW-SIZE       # window size [default: 256]
    --window WINDOW                 # window type [default: hann]
                                    #
    --learning-rate LEARNING-RATE   # learning rate [default: 1.0e-3]
    --weight-decay WEIGHT-DECAY     # weight decay [default: 1.0e-5]
    --factor FACTOR                 # factor [default: 0.7]
    --patience PATIENCE             # patience [default: 0]
    --min-learning-rate MIN-LEARNING-RATE
                                    # min-learning-rate [default: 1.0e-10]
    --clip-norm CLIP-NORM           # clip norm [default: 200]

Examples:
    ./train.py -d --epochs 2
"""

from docopt import docopt

import librosa

import torch as th
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

DEVICE = th.device("cuda:0" if th.cuda.is_available() else "cpu")

DEBUG = False
DEBUG_LEVEL = 1

def dprint(*args, **kwargs):
    """print if debug"""
    global DEBUG, DEBUG_LEVEL
    if DEBUG:
        level = 1
        # if caller supplied level kwarg then check it and delete
        if "level" in kwargs:
            level = kwargs["level"]
            # delete from kwargs sent to print
            del kwargs["level"]
        if DEBUG_LEVEL >= level:
            print(*args, **kwargs)


class UPITbLSTM(th.nn.Module):
    """Create a BLSTM module for uPIT"""
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
    """Train uPIT"""
    def __init__(self, args):
        self.hop_size = int(args['--hop-size'])
        self.window_size = int(args['--window-size'])
        self.window = args['--window']
        self.num_bins = fft_size(self.window_size) // 2 + 1

        dprint(f"Window size: {self.window_size}\n"
                f"Hop size: {self.hop_size}\n"
                f"Window: {self.window}")

        self.epochs = int(args['--epochs'])
        self.num_layers = int(args['--num-layers'])
        self.layer_size = int(args['--layer-size'])
        self.batch_size = int(args['--batch-size'])
        self.dropout = float(args['--dropout'])

        dprint(f"Epochs: {self.epochs}\n"
                f"Hidden layers: {self.num_layers}\n"
                f"Hidden layer size: {self.layer_size}\n"
                f"Dropout: {self.dropout}")

        self.lr = float(args['--learning-rate'])
        # self.clip_norm  = float(args['--clip-norm'])
        self.weight_decay = float(args['--weight-decay'])

        dprint(f"\nLearning rate: {self.lr}\nWeight Decay: {self.weight_decay}")

        self.factor = float(args['--factor'])
        self.patience = float(args['--patience'])
        self.min_lr = float(args['--min-learning-rate'])

        dprint(f"Factor: {self.factor}\n"
                f"Patience: {self.patience}\n"
                f"Min Learning Rate: {self.min_lr}")

        self.model = UPITbLSTM(self.num_bins,
                self.num_layers, self.layer_size, self.dropout)
        dprint(self.model)

        self.optimizer = th.optim.Adam(self.model.parameters(),
                lr=self.lr, weight_decay=self.weight_decay)
        self.sched = ReduceLROnPlateau(self.optimizer, mode='min',
                factor=self.factor,
                patience=self.patience,
                min_lr=self.min_lr,
                verbose=True)
        self.model.to(DEVICE)

    def train(self):
        self.model.train()
        total_loss = 0
        # Loop over dataset in batches
        for i in range(10):
            # get the input
            model_input = None

            self.optimizer.zero_grad()

            # compute the output masks
            masks = self.model(model_input)

            # compute the permutation loss

            # sum up the total loss

            # propagate error

            # clip norm

            self.optimizer.step()

        # return the average loss?
        return 1.0

    def run(self, train_data, valid_data, epochs):
        """Do a full training run across epochs."""
        for epoch in range(epochs):
            # check the time?
            train_loss = self.train(train_data)
            # do some validation
            val_loss = self.validate(val_data)
            self.sched.step(val_loss)

            # Save out the model params sometimes


from pathlib import Path

class WavLoader(object):
    """Load waves from disk applying the given transform

    w = WavLoader("train.txt", "data/2spk8kmax/tr/mix",
            stft, window_size=256, hop_size=128, window="hann")
    print([k.shape for k in w])

    """
    def __init__(self, listfile, rootdir, transform_fn=None, **tx_kwargs):
        # load the file list from listfile
        self.rootdir = Path(rootdir)
        self.transform_fn = transform_fn
        self.tx_kwargs = tx_kwargs

        with open(listfile, 'r') as f:
            self.files = f.readlines()

    def __iter__(self):
        for filename in self.files:
            full_fn = self.rootdir / filename.strip()
            dprint(f"WavLoader loading {full_fn}.")
            wav, _ = librosa.load(str(full_fn), sr=None)
            if self.transform_fn is not None:
                yield self.transform_fn(wav, **self.tx_kwargs)
            else:
                yield wav

def stft(wav, window_size, hop_size, window, take_abs=False, transpose=True):
    """Do an STFT with given options."""
    tf_mat = librosa.stft(wav, fft_size(window_size),
            hop_size, window_size, window=window)
    if take_abs:
        tf_mat = np.abs(tf_mat)
    if transpose:
        tf_mat = np.transpose(tf_mat)
    return tf_mat

def main(args):
    global DEBUG, DEBUG_LEVEL
    dprint(arguments)

    DEBUG = args['--debug']
    DEBUG_LEVEL = int(args['--debug-level'])

    # load training data
    # uttloader creates a Dataset which is passed to a DataLoader and returned
    # DataLoader implements __iter__(self) and yields the data in batches
    # load cv data

    train_filelist = args['--train-file-list']
    valid_filelist = args['--valid-file-list']

    train_data = WavLoader(train_filelist, "data/2spk8kmax/tr/mix",
            stft, window_size=256, hop_size=128, window="hann")
    valid_data = None

    # Bad design, but easy: read directly from args
    trainer = TrainUpit(args)
    trainer.run(train_data, valid_data)


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
