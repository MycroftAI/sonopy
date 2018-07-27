#!/usr/bin/env python3
# Copyright 2018 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from time import sleep, time

import numpy as np
from io import BytesIO

try:
    import requests
    import wavio
    from arriz import Arriz

    import speechpy
    import python_speech_features
    import librosa
    import sonopy

except ImportError as e:
    raise ValueError('Install extra dependencies with pip install sonopy[examples]: ' + str(e))

sample_rate = 16000
num_coeffs = 13
num_filts = 20
low_freq = 0
high_freq = 8000

# name: str, calc_mfcc: Callable, get_args: Callable
libraries = [
    ('Speechpy', speechpy.feature.mfcc, lambda signal, hop_t, fft_size: dict(
        signal=signal, sampling_frequency=sample_rate, frame_stride=hop_t, frame_length=hop_t,
        num_cepstral=num_coeffs, num_filters=num_filts, fft_length=fft_size, low_frequency=low_freq,
        high_frequency=high_freq, dc_elimination=True
    )),
    ('Python Speech Features', python_speech_features.mfcc,
     lambda signal, hop_t, fft_size: dict(
         signal=signal, samplerate=sample_rate, winlen=hop_t, winstep=hop_t, numcep=num_coeffs,
         nfilt=num_filts, nfft=fft_size, lowfreq=low_freq, highfreq=high_freq, preemph=0,
         ceplifter=0, appendEnergy=True
     )),
    ('Librosa', librosa.feature.mfcc, lambda signal, hop_t, fft_size: dict(
        y=signal, sr=sample_rate, hop_length=int(hop_t * sample_rate), n_mels=num_filts,
        n_mfcc=num_coeffs, n_fft=fft_size, fmin=low_freq, fmax=high_freq, norm=None
    )),
    ('Sonopy', sonopy.mfcc_spec, lambda signal, hop_t, fft_size: dict(
        audio=signal, sample_rate=sample_rate,
        window_stride=(int(hop_t * sample_rate), int(hop_t * sample_rate)),
        fft_size=fft_size, num_filt=num_filts, num_coeffs=num_coeffs
    ))
]

# length, hop_t, fft_size, loops
params = [
    (30 * sample_rate, 0.01, 2048, 20),
    (15 * sample_rate, 0.05, 2048, 200),
    (1 * sample_rate, 0.1, 2048, 2000),
    (1 * sample_rate, 0.1, 512, 20000),
]

demo_audio = 'https://raw.githubusercontent.com/MycroftAI/mycroft-core/' \
             '9ae13be1d46eea4f90db2b9377953101ce2139f9/test/unittests/client/data/record.wav'


def show_audio(url):
    signal = np.squeeze(wavio.read(BytesIO(requests.get(url).content)).data).astype('f')
    signal /= signal.max()
    for name, calc_mfcc, get_args in libraries:
        res = calc_mfcc(**get_args(signal, 0.01, 512))
        if len(res) < len(res[0]):
            res = np.rot90(res)
        Arriz.show(name, res[:, 1:])


def calculate_deltas(params):
    deltas = {}
    for i, (length, hop_t, fft_size, loops) in enumerate(params):
        print('===', chr(ord('A') + i), '===')
        signal = np.zeros(length)
        for name, calc_mfcc, get_args in libraries:
            begin = time()
            args = get_args(signal, hop_t, fft_size)
            for i in range(loops):
                calc_mfcc(**args)
            delta = time() - begin
            deltas.setdefault(name, []).append(delta)
            print(name + ':', delta)
    return deltas


def print_chart(lib_deltas):
    print()
    print('\t' + '\t'.join(chr(ord('A') + i) for i in range(len(params))))
    for name, deltas in lib_deltas.items():
        print('\t'.join(map(str, [name] + deltas)))

    print()
    print('\t'.join(['Param Set', 'Audio Len', 'Stride', 'Window', 'FFT Size', 'Sample Rate',
                     'Ceptral Coeffs', 'Num Filters', 'Loops']))
    for i, (length, hop_t, fft_size, loops) in enumerate(params):
        print('\t'.join(map(str, [
            chr(ord('A') + i), length, hop_t, hop_t, fft_size,
            sample_rate, num_coeffs, num_filts, loops
        ])))


def main():
    show_audio(demo_audio)
    lib_deltas = calculate_deltas(params)
    print_chart(lib_deltas)

    while Arriz.windows:
        sleep(0.5)


if __name__ == '__main__':
    main()
