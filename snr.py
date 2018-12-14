#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import scipy
import math
import random

import os.path

import glob

from pathlib import Path

import librosa

from sklearn.utils import shuffle
from sklearn.externals import joblib

def snr(s, s_hat):
    if len(s_hat) > len(s):
        s_hat = s_hat[:len(s)]
    if len(s) > len(s_hat):
        s = s[:len(s_hat)]
    s_diff = s - s_hat
    return 10 * np.log10(s.T.dot(s) / s_diff.T.dot(s_diff))

s1=Path("data/2spk8kmax/tt/s1")
s2=Path("data/2spk8kmax/tt/s2")

count = 0
total_snr = 0

min_snr = math.inf
max_snr = -math.inf

for f in glob.glob('./rel1000/*.wav'):
    fp = Path(f)
    # print(fp.name)
    sep_name = fp.name
    if "spk2" in sep_name:
        continue
    i = sep_name.index(".spk")
    orig_name = sep_name[:i] + ".wav"
    shat1_full = str(fp)
    shat1_full_i = shat1_full.index(".spk")
    shat2_full = shat1_full[:shat1_full_i] + ".spk2.wav"
    s1w, _ = librosa.load(str(s1/orig_name), sr=None)
    s2w, _ = librosa.load(str(s2/orig_name), sr=None)
    s1_hat, _ = librosa.load(str(fp), sr=None)
    s2_hat, _ = librosa.load(shat2_full, sr=None)
    snr_p11_res = snr(s1w, s1_hat)
    snr_p12_res = snr(s1w, s2_hat)
    snr_p21_res = snr(s2w, s1_hat)
    snr_p22_res = snr(s2w, s2_hat)
    if snr_p11_res + snr_p22_res > snr_p12_res + snr_p21_res:
        print(f"{orig_name} SNR1: {snr_p11_res} SNR2: {snr_p22_res}")
        total_snr += snr_p11_res + snr_p22_res
        min_snr = min(min_snr, snr_p11_res, snr_p22_res)
        max_snr = max(max_snr, snr_p11_res, snr_p22_res)
    else:
        print(f"{orig_name} SNR1: {snr_p12_res} SNR2: {snr_p21_res}")
        total_snr += snr_p12_res + snr_p21_res
        min_snr = min(min_snr, snr_p12_res, snr_p21_res)
        max_snr = max(max_snr, snr_p12_res, snr_p21_res)
    count += 2

print(f"Min SNR: {min_snr} Max SNR: {max_snr}")
print(f"Average SNR: {total_snr/count}")


