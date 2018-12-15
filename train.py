#!/usr/bin/env python3
# train.py
"""Train a uPIT BLSTM

Usage:
    train.py [options]

Options:
    -d, --debug                         # output extra debug information
    --debug-level LEVEL                 # debug level to LEVEL [default: 1]
                                        #
    --train-root-dir TRAIN-ROOT-DIR     # training set root directory [default: data/2spk8kmax/tr]
    --valid-root-dir VALID-ROOT-DIR     # validation set root directory [default: data/2spk8kmax/cv]
    --mix-dir MIX-DIR                   # directory beneath root with mixtures [default: mix]
    --src1-dir SRC1-DIR                 # directory beneath root with source 1 [default: s1]
    --src2-dir SRC2-DIR                 # directory beneath root with source 2 [default: s2]
    --train-file-list TRAIN-FL          # file that contains training set [default: train.txt]
    --valid-file-list VAL-FL            # file that contains validation set [default: validation.txt]
    --output-dir OUTPUT-DIR             # output directory [default: out]
    --save-every SAVE-EVERY             # save model every N epochs [default: 10]
    --init-model INIT-MODEL             # load initial model parameters
                                        #
    --num-speakers NUM-SPEAKERS         # number of speakers [default: 2]
    --epochs EPOCHS                     # number of epochs [default: 10]
    --batch-size BATCH-SIZE             # batch size [default: 8]
    --num-layers NUM-LAYERS             # number of hidden layers [default: 3]
    --layer-size LAYER-SIZE             # size of each hidden layer [default: 896]
    --dropout DROPOUT                   # dropout rate [default: 0.5]
                                        #
    --hop-size HOP-SIZE                 # hop size [default: 128]
    --window-size WINDOW-SIZE           # window size [default: 256]
    --window WINDOW                     # window type [default: hann]
                                        #
    --learning-rate LEARNING-RATE       # learning rate [default: 1.0e-3]
    --weight-decay WEIGHT-DECAY         # weight decay [default: 1.0e-5]
    --factor FACTOR                     # factor [default: 0.7]
    --patience PATIENCE                 # patience [default: 0]
    --min-learning-rate MIN-LEARNING-RATE
                                        # min-learning-rate [default: 1.0e-10]
    --clip-norm CLIP-NORM               # clip norm [default: 200]

Examples:
    ./train.py -d --epochs 2
    ./train.py -d --debug-level 3 --epochs 2  --batch-size 2
"""

from docopt import docopt

from itertools import permutations
import time

from pathlib import Path

import librosa

import numpy as np

import torch as th
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import torch.nn.functional as F

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
    def __init__(self, num_bins, num_s=2,
            num_layers=3, layer_size=896, dropout=0.5):
        super(UPITbLSTM, self).__init__()
        self.num_s = num_s
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
        self.num_s = int(args['--num-speakers'])

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
        self.clip_norm  = float(args['--clip-norm'])
        self.weight_decay = float(args['--weight-decay'])

        dprint(f"\nLearning rate: {self.lr}\nWeight Decay: {self.weight_decay}")

        self.factor = float(args['--factor'])
        self.patience = float(args['--patience'])
        self.min_lr = float(args['--min-learning-rate'])

        dprint(f"Factor: {self.factor}\n"
                f"Patience: {self.patience}\n"
                f"Min Learning Rate: {self.min_lr}")

        self.model = UPITbLSTM(self.num_bins, self.num_s,
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

    def validate(self, valid_data):
        self.model.eval()

        total_loss = 0
        batch_count = 0
        # Loop over dataset in batches
        n = len(valid_data)
        indices = list(range(n))
        np.random.shuffle(indices)
        for batch_i in range(0, n, self.batch_size):
            batch_count += 1
            batch_indices = indices[batch_i:batch_i+self.batch_size]
            # Sort longest first to be able to pack
            # train_data returns (mix_spec, [src1_spec, src2_spec])
            batch_set = sorted([valid_data[i] for i in batch_indices],
                    key=lambda x: x[0].shape[0],
                    reverse=True)
            dprint(f"Batch indices: {batch_indices}", level=2)

            input_sizes = th.tensor([mix.shape[0] for (mix, _) in batch_set],
                    dtype=th.float32)
            model_input = pack_sequence([
                th.tensor(np.abs(mix), dtype=th.float32)
                for (mix, _) in batch_set])

            mix_spec = pad_sequence([
                th.tensor(np.abs(mix), dtype=th.float32)
                for (mix, _) in batch_set], batch_first=True)

            mix_phase = pad_sequence([
                th.tensor(np.angle(mix), dtype=th.float32)
                for (mix, _) in batch_set], batch_first=True)

            sources_spec = []
            for i in range(self.num_s):
                source_Ns_spec = pad_sequence([
                    th.tensor(np.abs(sources[i]), dtype=th.float32)
                    for (_, sources) in batch_set], batch_first=True)
                sources_spec.append(source_Ns_spec)

            sources_phase = []
            for i in range(self.num_s):
                source_Ns_phase = pad_sequence([
                    th.tensor(np.abs(sources[i]), dtype=th.float32)
                    for (_, sources) in batch_set], batch_first=True)
                sources_phase.append(source_Ns_phase)

            model_input = (model_input.cuda() if th.cuda.is_available() else
                    model_input.to(DEVICE))

            dprint(f"  input_sizes: {input_sizes.size()}", level=3)
            dprint(f"  model_input: {type(model_input)}", level=3)
            # compute the output masks
            est_masks = self.model(model_input)

            dprint(f"  est_masks: {type(est_masks)} {len(est_masks)}", level=3)

            # compute the permutation loss
            batch_loss = self.total_permutation_loss(est_masks,
                    input_sizes, mix_spec, mix_phase,
                    sources_spec, sources_phase)

            # sum up the total loss
            total_loss += batch_loss.item()

        # return the average loss?
        return total_loss / batch_count

    def train(self, train_data):
        self.model.train()
        total_loss = 0
        batch_count = 0
        # Loop over dataset in batches
        n = len(train_data)
        indices = list(range(n))
        np.random.shuffle(indices)
        for batch_i in range(0, n, self.batch_size):
            batch_count += 1
            batch_indices = indices[batch_i:batch_i+self.batch_size]
            # Sort longest first to be able to pack
            # train_data returns (mix_spec, [src1_spec, src2_spec])
            batch_set = sorted([train_data[i] for i in batch_indices],
                    key=lambda x: x[0].shape[0],
                    reverse=True)
            dprint(f"Batch indices: {batch_indices}", level=2)
            if DEBUG_LEVEL >= 3:
                for i, (mix, sources) in enumerate(batch_set):
                    dprint(f"  {i:2d} Mix: {mix.shape}, sources: {[s.shape for s in sources]}", level=3)

            # DataLoader._transform called on dataset output one at a time
            #   (mix_spec, [src1_spec, src2_spec])
            # And it returns:
            #   num_frames: mix_spec.shape[0]
            #   feature: th.tensor(np.abs(mix_spec), dtype=th.float32)
            #   source_attr:
            #       spectrogram: th.tensor(np.abs(mix_spec), dtype=th.float32)
            #       phase: th.tensor(np.angle(mix_spec), dtype=th.float32)
            #   target_attr:
            #       spectrogram: [th.tensor(np.abs(mix_spec), dtype=th.float32),
            #           ...]
            #       phase: [th.tensor(np.angle(mix_spec), dtype=th.float32),
            #           ...]

            input_sizes = th.tensor([mix.shape[0] for (mix, _) in batch_set],
                    dtype=th.float32)
            model_input = pack_sequence([
                th.tensor(np.abs(mix), dtype=th.float32)
                for (mix, _) in batch_set])

            mix_spec = pad_sequence([
                th.tensor(np.abs(mix), dtype=th.float32)
                for (mix, _) in batch_set], batch_first=True)

            mix_phase = pad_sequence([
                th.tensor(np.angle(mix), dtype=th.float32)
                for (mix, _) in batch_set], batch_first=True)

            sources_spec = []
            for i in range(self.num_s):
                source_Ns_spec = pad_sequence([
                    th.tensor(np.abs(sources[i]), dtype=th.float32)
                    for (_, sources) in batch_set], batch_first=True)
                sources_spec.append(source_Ns_spec)

            sources_phase = []
            for i in range(self.num_s):
                source_Ns_phase = pad_sequence([
                    th.tensor(np.abs(sources[i]), dtype=th.float32)
                    for (_, sources) in batch_set], batch_first=True)
                sources_phase.append(source_Ns_phase)

            # This was wrong: it should be a pad_sequence across the sources,
            # not a pad_sequence per source.
            #sources_spec = [pad_sequence([
                #th.tensor(np.abs(source), dtype=th.float32)
                #for source in sources], batch_first=True)
                #for (_, sources) in batch_set]
            #sources_phase = [pad_sequence([
                #th.tensor(np.angle(source), dtype=th.float32)
                #for source in sources], batch_first=True)
                #for (_, sources) in batch_set]

            if DEBUG_LEVEL >= 3:
                for i, src in enumerate(sources_spec):
                    dprint(f" source_spec {i:3d}: {src.size()}", level=3)
                for i, src in enumerate(sources_phase):
                    dprint(f" source_phase {i:3d}: {src.size()}", level=3)
            dprint(f"sources_spec {len(sources_spec)}", level=3)
            dprint(f"sources_phase {len(sources_phase)}", level=3)

            # DataLoader._process(batch)
            #   * calls _transform to get a list of dicts
            #   * sorts by reverse num_frames (longest first...)
            # Returns:
            #   input_sizes: th.tensor([num_frames in batch], dtype=th.float32)
            #   input_feats: pack_squence([feature in batch])
            #   source_attr:
            #       spectrogram: pad_sequence([spectrogram in batch],
            #           batch_first=True)
            #       phase: pad_sequence([phase in batch],
            #           batch_first=True)
            #   target_attr:
            #       spectrogram: [pad_sequence([spectrogram in target],
            #           batch_first=True), ...]
            #       phase: [pad_sequence([phase in target],
            #           batch_first=True), ...]


            # get the input
            # model_input = packed_sequence_cuda(model_input) if isinstance(
                # model_input, PackedSequence) else model_input.to(device)
            model_input = (model_input.cuda() if th.cuda.is_available() else
                    model_input.to(DEVICE))

            self.optimizer.zero_grad()

            dprint(f"  input_sizes: {input_sizes.size()}", level=3)
            dprint(f"  model_input: {type(model_input)}", level=3)
            # compute the output masks
            est_masks = self.model(model_input)

            dprint(f"  est_masks: {type(est_masks)} {len(est_masks)}", level=3)

            # compute the permutation loss
            batch_loss = self.total_permutation_loss(est_masks,
                    input_sizes, mix_spec, mix_phase,
                    sources_spec, sources_phase)

            # sum up the total loss
            total_loss += batch_loss.item()

            # propagate error
            batch_loss.backward()

            # clip norm
            if self.clip_norm:
                th.nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.clip_norm)

            self.optimizer.step()

        # return the average loss?
        return total_loss / batch_count

    def total_permutation_loss(self, est_masks, input_sizes,
            mix_spec, mix_phase,
            sources_spec, sources_phase):
        """Calculate the permutation loss for backprop"""
        input_sizes = input_sizes.to(DEVICE)
        mix_spec = mix_spec.to(DEVICE)
        mix_phase = mix_phase.to(DEVICE)
        sources_spec = [src.to(DEVICE) for src in sources_spec]
        sources_phase = [src.to(DEVICE) for src in sources_phase]

        def per_permutation_loss(permutation):
            """calc loss for each permutation of sources"""
            dprint(f"per_permutation_loss()", level=3)
            loss = []
            for mix_i, src_i in enumerate(permutation):
                dprint(f"  permute: {mix_i}, {src_i}", level=3)
                dprint(f"  mix_phase: {mix_phase.size()}", level=3)
                dprint(f"  sources_spec: {sources_spec[src_i].size()}", level=3)
                dprint(f"  sources_phase: {sources_phase[src_i].size()}",
                        level=3)
                #spect = sources_spec[src_i]
                spect = sources_spec[src_i] * F.relu(
                        th.cos(mix_phase - sources_phase[src_i]))
                dprint(f"  spect: {spect.size()}", level=3)
                dprint(f"  mix_spec: {mix_spec.size()}", level=3)
                dprint(f"  est_masks: {est_masks[mix_i].size()}", level=3)
                full_utt_loss = th.sum(th.sum(
                            th.pow(est_masks[mix_i] * mix_spec - spect, 2),
                            -1), -1)
                loss.append(full_utt_loss)
            this_loss = sum(loss) / input_sizes
            return this_loss

        batch_size = input_sizes.shape[0]
        permute_losses = th.stack(
                [per_permutation_loss(p)
                    for p in permutations(range(self.num_s))])
        min_loss, _ = th.min(permute_losses, dim=0)
        return th.sum(min_loss) / (self.num_s * batch_size)

    def run(self, args):
        """Do a full training run across epochs."""
        # load training data
        # uttloader creates a Dataset which is passed to a DataLoader and returned
        # DataLoader implements __iter__(self) and yields the data in batches
        # load cv data

        dprint(f"Beginning training on {DEVICE}...")

        self.out_dir = Path(args['--output-dir'])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(args['--save-every'])

        train_filelist = args['--train-file-list']
        valid_filelist = args['--valid-file-list']
        train_root = Path(args['--train-root-dir'])
        valid_root = Path(args['--valid-root-dir'])
        mix_dir = Path(args['--mix-dir'])
        s1_dir = Path(args['--src1-dir'])
        s2_dir = Path(args['--src2-dir'])

        train_data = MixSourceLoader(
            WavLoader(train_filelist, train_root / mix_dir,
                stft, window_size=self.window_size, take_abs=True,
                hop_size=self.hop_size, window=self.window),
            [
                WavLoader(train_filelist, train_root / s1_dir,
                    stft, window_size=self.window_size, take_abs=True,
                    hop_size=self.hop_size, window=self.window),
                WavLoader(train_filelist, train_root / s2_dir,
                    stft, window_size=self.window_size, take_abs=True,
                    hop_size=self.hop_size, window=self.window),
            ])
        valid_data = MixSourceLoader(
            WavLoader(valid_filelist, valid_root / mix_dir,
                stft, window_size=self.window_size, take_abs=True,
                hop_size=self.hop_size, window=self.window),
            [
                WavLoader(valid_filelist, valid_root / s1_dir,
                    stft, window_size=self.window_size, take_abs=True,
                    hop_size=self.hop_size, window=self.window),
                WavLoader(valid_filelist, valid_root / s2_dir,
                    stft, window_size=self.window_size, take_abs=True,
                    hop_size=self.hop_size, window=self.window),
            ])

        for epoch in range(self.epochs):
            dprint(f"Epoch {epoch:3d}.")

            dprint("  Begin training...")
            train_start = time.time()
            train_loss = self.train(train_data)

            dprint("  Validation...")
            val_start = time.time()
            val_loss = self.validate(valid_data)
            val_end = time.time()

            # step the lr scheduler based on validation loss
            self.sched.step(val_loss)

            dprint(f"  training loss: {train_loss:.4f}"
                    f"({val_start-train_start:.1f}s) val loss: {val_loss:.4f} "
                    f"({val_end-val_start:.1f}s)")

            # Save out the model params sometimes
            if (epoch+1) % self.save_every == 0 or (epoch+1) == self.epochs:
                save_path = self.out_dir / f"model.{epoch+1:02d}.pkl"
                dprint(f"Saving model to {save_path}")
                th.save(self.model.state_dict(), save_path)


class WavLoader(object):
    """Load waves from disk applying the given transform

    To disable the caching behavior set self._cache to None

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

        # Enable or disable cache by default
        # self._cache = {}
        self._cache = None

    def _load(self, filename):
        full_fn = self.rootdir / filename.strip()
        dprint(f"WavLoader loading {full_fn}.", level=3)
        wav, _ = librosa.load(str(full_fn), sr=None)
        if self.transform_fn is not None:
            return self.transform_fn(wav, **self.tx_kwargs)
        else:
            return wav

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        if self._cache is not None:
            if key not in self._cache:
                self._cache[key] = self._load(self.files[key])
            return self._cache[key]
        else:
            return self._load(self.files[key])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

def stft(wav, window_size, hop_size, window, take_abs=False, transpose=True):
    """Do an STFT with given options."""
    tf_mat = librosa.stft(wav, fft_size(window_size),
            hop_size, window_size, window=window)
    if take_abs:
        tf_mat = np.abs(tf_mat)
    if transpose:
        tf_mat = np.transpose(tf_mat)
    return tf_mat

class MixSourceLoader(object):
    def __init__(self, mix_loader, source_loaders):
        self.mix_loader = mix_loader
        self.source_loaders = source_loaders

    def __len__(self):
        return len(self.mix_loader)

    def __getitem__(self, key):
        mix = self.mix_loader[key]
        sources = [loader[key] for loader in self.source_loaders]
        return (mix, sources)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def main(args):
    global DEBUG, DEBUG_LEVEL

    DEBUG = args['--debug']
    DEBUG_LEVEL = int(args['--debug-level'])

    dprint(arguments)

    # Bad design, but easy: read directly from args
    trainer = TrainUpit(args)
    trainer.run(args)
    dprint("Training Complete")


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
