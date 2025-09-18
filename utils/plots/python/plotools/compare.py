# ===- compare.py ---------------------------------------------------------===

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===--------------------------------------------------------------------===

# This file contains the main functions used to plot the figures.

# ===--------------------------------------------------------------------===

import wave
from typing import Tuple, Dict, Union
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from scipy.io import wavfile
import matplotlib.gridspec as gridspec


def get_info_and_samples(file: str) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Parse the .wav file to get some basic information and the sample data.
    :param file: The path of the .wav file.
    :return: A dictionary that contains basic information and a numpy array
             that contains the sample data.
    """
    # get basic information of the .wav file
    with wave.open(file, 'rb') as audio:
        # get the number of channels, i.e. 1 for mono and 2 or stereo
        num_channels = audio.getnchannels()
        # get the total number of frames in the file
        num_frames = audio.getnframes()
        # get sample width in bytes
        samp_width = audio.getsampwidth()

    # read frame data
    framerate, samples = wavfile.read(file)

    return {'num_channels': num_channels, 'num_frames': num_frames,
            'sample_width': samp_width, 'framerate': framerate}, samples


def get_time_domain(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the necessary arrays needed to plot the figure of time domain.
    :param file: The path of the .wav file.
    :return: The sample data array (y-axis) and the time slot array (x-axis).
    """
    # get basic information and samples data
    info, samples = get_info_and_samples(file)
    num_channels = info['num_channels']
    samp_width = info['sample_width']
    num_frames = info['num_frames']
    framerate = info['framerate']

    # with stereo, we only reserve the left channel for display
    if num_channels > 1:
        samples = np.transpose(samples)[0]

    # normalize: element / max_value
    byte = 2 ** np.ceil(np.log(samp_width) / np.log(2))
    samples = samples / (2 ** (8 * byte - 1) - 1)

    # calculate the time(s) used for labeling the x-axis
    #       total_time = num_frames / framerate              (s)
    time = np.arange(0, num_frames) / framerate

    return samples, time


def get_frequency_domain(file: str):
    """
    Get the necessary arrays needed to plot the figure of frequency domain.
    :param file: The path of the .wav file.
    :return: The power data after FFT on sample data array (y-axis) and the
             frequency array (x-axis).
    """
    # get basic information and samples data
    info, samples = get_info_and_samples(file)
    framerate = info['framerate']

    # get power of the wave signal
    fft_complex = rfft(samples)
    power = np.abs(fft_complex)

    # get frequency array
    freq = rfftfreq(len(samples), 1 / framerate)

    return power, freq


def compare_wave(file1: str, file2: str = None, part: str = "body",
                 granularity: Union[int, float] = 5000, save: bool = True):
    """
    Plot the comparison figures between before and after the process in
    time-domain and frequency-domain.
    :param file1: The path of the .wav file before process.
    :param file2: The path of the .wav file after process. If no file2 passed
                  in, we will only plot waves of file1.
    :param part: Choose a part of the time-domain figure to show. Possible
                 values are "head", "body", "tail" and "all".
    :param granularity: Choose how many points is to be plotted. It should be
                        either a portion(float between 0~1) or a number(int).
    :param save: A bool value to decide whether to save the figure.
    :return: None.
    """
    # plot time-domain figure
    before, time1 = get_time_domain(file1)
    if file2 is not None:
        after, time2 = get_time_domain(file2)
        # As we aim to compare signals between before and after the process, we
        # have to promise that the time slot arrays remain the same.
        assert (time1 == time2).all(), "Time slot arrays are not equal!"

    # As the time-domain figure is usually too large to see the difference, we
    # provide customized options to let the user decide which part should be
    # plotted.
    # calculate the number of points to be plotted
    N = len(time1)
    # the granularity could be portion
    if 0 <= granularity <= 1:
        n = int(N * granularity)
        assert n > 0, "Few points to plot!"
    # the granularity could be number of points to be plotted
    elif granularity % 1 == 0:
        n = int(granularity)
    else:
        raise ValueError(
            "Parameter 'granularity' should be either a "
            "portion(float between 0~1) or a number(int)!")
    # calculate the region to be plotted
    if part == "head":
        start = 0
        end = n
    elif part == "body":
        start = int((N - n) / 2)
        end = start + n
    elif part == "tail":
        start = N - n
        end = N
    elif part == "all":
        start = 0
        end = N
    before = before[start:end]
    time1 = time1[start:end]

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(time1, before, linewidth=1, label="before")
    if file2 is not None:
        after = after[start:end]
        time2 = time2[start:end]
        ax.plot(time2, after, linewidth=1, label="after")
        ax.legend()
    ax.set_title("Time Domain")

    # plot time-domain figure
    # We reserve the whole plot to show the difference of the high-frequency
    # part.
    p_before, freq1 = get_frequency_domain(file1)

    ax = fig.add_subplot(gs[1, :])
    ax.semilogy(freq1, p_before, label="before")
    if file2 is not None:
        p_after, freq2 = get_frequency_domain(file2)
        assert (freq1 == freq2).all(), "Frequency arrays are not equal!"
        ax.semilogy(freq2, p_after, color="orange", label="after")
        ax.legend()
    ax.set_title("Frequency Domain")

    fig.align_labels()

    if save:
        plt.savefig("res.png", dpi=200)
    plt.show()
