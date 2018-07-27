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
from functools import partial
from pylisten import FeatureListener

from arriz import Arriz
from sonopy import mel_spec


def main():
    sample_rate = 16000
    stride = 500
    width = 60
    processor = partial(
        mel_spec, sample_rate=sample_rate, window_stride=(2 * stride, stride),
        fft_size=512, num_filt=20
    )
    for features in FeatureListener(processor, stride, width, dict(rate=sample_rate)):
        if not Arriz.show('Mel spectrogram', features):
            break


if __name__ == '__main__':
    main()
