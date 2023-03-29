"""
Docstring: 
"""
import functools
from enum import Enum
from typing import Union, NewType
import os
import wave
import numpy as np
import numpy.typing as npt
import sounddevice as sd
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

MIN_HUMAN_FREQ: float = 80
MAX_HUMAN_FREQ: float = 20000


@functools.partial(np.vectorize, excluded=[1, 2])
def linear_interpolation(x: npt.ArrayLike, known_xs: npt.NDArray, known_ys: npt.NDArray,
                         xs_sorted: bool = False, allow_out_of_bounds: bool = True):
    #region compress DocString and error strings
    """
    Find the value at x of linear interpolation of known points (known_xs, known_ys).
        x: arrayLike
        known_xs: array
        known_ys: array
        xs_sorted: bool = False
            > is known_xs sorted?
        allow_out_of_bounds: bool
            > should interpolation outside the bounds of the known values be allowed or throw an error.
    """

    # Defines error messages
    too_many_dimensions_error = \
        "this linear interpolation function only works on Functions with one dimensional input."
    non_matching_dimensions_error = "known_xs and known_ys must be of matching lenght"
    out_of_bounds_error = "Can only interpolate within the bounds of the function."
    two_small_error = "must have at least two sample points"
    #endregion

    # Check inputs are correct
    assert len(known_xs.shape) == 1, too_many_dimensions_error
    assert len(known_ys.shape) == 1, too_many_dimensions_error
    assert known_xs.size == known_ys.size, non_matching_dimensions_error
    assert known_xs.size > 1, two_small_error
    if allow_out_of_bounds == False:
        assert known_xs[0] < x < known_xs[-1], out_of_bounds_error

    # Sort the known values if they are unsorted
    if xs_sorted == False:
        sorted_indicies = np.argsort(known_xs)
        known_xs = known_xs
        known_ys = known_ys

    # get the index of the two points that define the line segment used for interpolation.
    if x < known_xs[0]:
        i = 0
    elif x > known_xs[-1]:
        i = known_xs.size-2
    else:
        i = known_xs.searchsorted(x)-1
    
    # Work out the slope and offset of the line
    m = (known_ys[i+1]-known_ys[i])/(known_xs[i+1]-known_xs[i])
    c = known_ys[i]-m*known_xs[i]

    # Find the predicted value using the equation for the line
    return m*x + c


def mels_to_hzs(mels: npt.ArrayLike):
    """Takes an array of mels values and returns an array of hz"""
    return 700.0 * (10**(mels / 2595.0) - 1.0)


def hzs_to_mels(hzs: npt.ArrayLike):
    """Takes an array of mels values and returns an array of hz"""
    return 2595.0 * np.log10(1.0 + hzs / 700.0)


class FrequencyScale(Enum):
    """
    Different ways to scale frequencies
    """
    LINEAR = {"transform": lambda f: f, "inverse": lambda scale: scale}
    LOG2 = {"transform": np.log2, "inverse": np.exp2}
    MEL = {"transform": hzs_to_mels, "inverse": mels_to_hzs}


FrequencyScales = NewType('FrequencyScales', FrequencyScale)


def human_scale_array(
    num_samples: int,
    scale: FrequencyScales = FrequencyScale.LOG2
) -> npt.NDArray:
    """
    Returns an equally spaced array of num_samples throughout the range humans can hear.
    The array is in the specified scale, for instance if scale = MEL, the results are in
    the given scale and equally spaced in the given scale.
    """
    min_scaled = scale.value["transform"](MIN_HUMAN_FREQ)
    max_scaled = scale.value["transform"](MAX_HUMAN_FREQ)
    increment = (max_scaled-min_scaled)/num_samples
    return np.arange(min_scaled, max_scaled, increment)


def human_frequencies_array(
    num_samples: int,
    scale: FrequencyScales = FrequencyScale.LOG2
) -> npt.NDArray:
    """
    Returns an array of num_samples frequencies that humans can hear.
    Spaced evenly in scale given in the scale parameter.
    """
    return scale.value["inverse"](human_scale_array(num_samples, scale))


def resample_forrier_strengths(
    array: npt.NDArray,
    num_samples: int,
    sample_rate: int,
    scale: FrequencyScales
) -> npt.NDArray:
    """
    Takes array produced by np.fft.fft and returns what the frequency strength should be
    at various frequencies using linear interpolation.
    """
    return linear_interpolation(
        human_frequencies_array(num_samples, scale),
        np.arange(len(array))*sample_rate/len(array),
        array
    )


def forrier_pad(array: npt.NDArray) -> npt.NDArray:
    """
    pads an array with 0s such that it's length is of power 2.
    Useful for FFT.
    """
    len_to_round_to = 2**np.ceil(np.log2(len(array)))
    to_pad = int(len_to_round_to - len(array))
    return np.pad(array, (0, to_pad))


def fast_fourier_transform(array: npt.NDArray, sanitized: bool = False) -> npt.NDArray:
    """
    FFT for a one dimensional array
    will pad input with zeroes till it reaches power of 2 if sanitzed == False.
    This function is not used as np.fft.fft is more optimised and runs better on
    arrays of variable length.
    """
    # pads end of array with zeroes so array has a length which is a power of 2
    if not sanitized:
        array = forrier_pad(array)

    assert (2**np.ceil(np.log2(len(array))) == (len(array)))

    length = len(array)
    if length == 1:
        return array

    transform_even = fast_fourier_transform(array[::2])
    transform_odd = fast_fourier_transform(array[1::2])

    theta = np.exp(-2j*np.pi*1/length)

    return np.concatenate(
        [transform_even + np.power(theta, np.arange(length/2)) * transform_odd,
         transform_even + np.power(theta, np.arange(length/2, length)) * transform_odd])


fft = np.fft.fft

def calculate_spectrogram(array: npt.NDArray, sample_rate: int, window: float, overlap: float = 0.1,
                              scale: FrequencyScales = FrequencyScale.LOG2) -> npt.NDArray:
    """
    Calculates a spectogram from an array.
    window: float > how long of a window to split the audio into before taking DFT
    overlap: how much overlap between windows as a proportion of the window size.
    """
    increment = window*(1-overlap)
    window_frames = int(window*sample_rate)
    n_windows = -1 + len(array) / (sample_rate * increment)
    window_ts = np.arange(0, n_windows) * increment * sample_rate

    windows = np.lib.stride_tricks.sliding_window_view(
        array, window_frames)[window_ts.astype(int)]

    spectrogram = fft(windows)
    spectrogram = np.apply_along_axis(func1d=resample_forrier_strengths, axis=1, arr=spectrogram,
                                    num_samples=400, sample_rate = sample_rate, scale=scale)

    return spectrogram, window_ts

def show_spectrogram(values: npt.NDArray, times: npt.NDArray, sample_rate: int, plot=plt,
                     scale: FrequencyScales = FrequencyScale.LOG2,
                     display_with_scale: bool = False):
    "Plots a spectogram of the audio"

    values = values.T
    if display_with_scale:
        y_scale = human_scale_array(num_samples=400, scale=scale)
        y_scale = [int(y) for y in y_scale]
    else:
        y_scale = human_frequencies_array(num_samples=400, scale=scale)
        y_scale = [int(y) for y in y_scale]

    times = [f'{x:.2f}' for x in times/sample_rate]

    df = pd.DataFrame(np.absolute(values), index=y_scale, columns=times)

    sns.heatmap(df)
    plot.show()


class DiscreteAudio:
    """
    Class that encapsulates audio data stored in a np array.
    Can be used to plot the data or analyse/plot it's spectrogram. 
    """

    def __init__(self, audio_array, sample_rate):
        self.array = audio_array
        self.sample_rate = sample_rate

    @property
    def num_channels(self):
        "returns the number of channels in the audio"
        if len(self.array.shape) < 2:
            return 1
        return self.array.shape[1]

    @property
    def mono(self) -> npt.NDArray:
        "Returns the mono (single channel) representation of the Object's audio as a 1D np array."
        if len(self.array.shape) == 1:
            return self.array
        return np.mean(self.array, axis=1)

    def play_mono(self):
        """
        plays the object's audio as mono.
        """
        sd.wait()
        sd.play(self.mono, self.sample_rate)
        sd.wait()

    def play(self):
        "plays the object's audio"

        # Needs both the waits otherwise the python script will close before
        # the audio plays in full
        sd.wait()
        sd.play(self.array)
        sd.wait()

    def plot_mono(self, show: bool = True):
        "plots the object's mono audio to a plt graph"
        plt.plot(np.arange(len(self.mono))/self.sample_rate, self.mono)
        if show:
            plt.show()

    def plot(self, show: bool = True):
        "plots the object's audio to a plt graph"
        if self.num_channels > 1:
            for channel in range(np.size(self.array, 1)):
                plt.plot(np.arange(len(self.mono)) /
                         self.sample_rate, self.array[:, channel])
            if show:
                plt.show()
        else:
            self.plot_mono(show=show)

    def write(self, file: Union[str, bytes, os.PathLike]):
        "Write object audio to the specified filepath."
        if not file.endswith(".wav"):
            file += ".wav"

        wave_array = (self.array*2**31).astype(np.int32)

        with wave.open(file, mode="wb") as wave_file:
            wave_file: wave.Wave_write

            wave_file.setparams(
                (self.num_channels, 4, self.sample_rate, len(self.mono), "NONE", "NONE"))
            wave_file.writeframesraw(wave_array.tobytes())

    def calculate_spectrogram(self, window: float, overlap: float = 0.1,
                              scale: FrequencyScales = FrequencyScale.LOG2) -> npt.NDArray:
        """
        Calculates a spectogram from the object's mono audio.
        window: float > how long of a window to split the audio into before taking DFT
        overlap: how much overlap between windows as a proportion of the window size.
        """
        return calculate_spectrogram(self.mono, self.sample_rate, window, overlap, scale)

    def show_spectrogram(self, plot=plt, scale: FrequencyScales = FrequencyScale.LOG2,
                         display_with_scale: bool = False):
        "Plots a spectogram of the audio"
        values, times = self.calculate_spectrogram(0.1, overlap=0.1, scale = scale)

        values = values.T
        if display_with_scale:
            y_scale = human_scale_array(num_samples=400, scale=scale)
            y_scale = [int(y) for y in y_scale]
        else:
            y_scale = human_frequencies_array(num_samples=400, scale=scale)
            y_scale = [int(y) for y in y_scale]

        times = [f'{x:.2f}' for x in times/self.sample_rate]
        df = pd.DataFrame(np.absolute(values), index=y_scale, columns=times)

        sns.heatmap(df)
        plot.show()


def read_wave_file(file: Union[str, bytes, os.PathLike]):
    """
    Open's .wav file and return's a DiscreteAudio object.
    DiscreteAudio object will have a maxiumum possible amplitude of 1. 
    Takes a filepath as an input.
    Accuratly reads audio with multiple channels.
    """

    # Get's file's properties using the wave module of the core library.
    # Also reads the audio's bytes to a byte Array.
    with wave.open(file, mode="rb") as wave_file:
        n_frames = wave_file.getnframes()
        audio_bytes = wave_file.readframes(n_frames)
        sample_width = wave_file.getsampwidth()
        n_channels = wave_file.getnchannels()
        sample_rate = wave_file.getframerate()

    # Creates an empty 2D array in which to store the audio.
    # Each row of the array corresponds to a different audio channel.
    audio_array = np.zeros((n_frames, n_channels))

    # Loop over each frame. For each frame loop over each audio channel.
    # For each frame and audio channel set the corresponding element in
    # audioArray to the value in the .wav file
    for i in range(0, n_frames):
        for j in range(n_channels):
            audio_array[i, j] = int.from_bytes(
                audio_bytes[(i*n_channels+j)*sample_width:(i *
                                                           n_channels+1+j)*sample_width],
                byteorder='little',
                signed=True
            )
    # Scale the audioArray so the max amplitude is 1.
    audio_array /= 2**(8*sample_width)
    # Return a DiscreteAudio object intialised with audioArray
    return DiscreteAudio(audio_array, sample_rate)


def main():
    """Run sanity tests"""

    drum_audio = read_wave_file("stereo_test.wav")
    drum_audio.plot()
    drum_audio.show_spectrogram(
        scale=FrequencyScale.MEL, display_with_scale=False)
    drum_audio.play()
    drum_audio.write("test_output.wav")


if __name__ == "__main__":
    main()

# print(drumAudio.calculate_spectrogram(0.1).shape)

# drumAudio.play()

# drumAudio.write("out")
# drumAudio.plot()

# print(len(np.fft.fft(drumAudio.array[:,0])), len(fastFourierTransform(drumAudio.array[:,0])))

# paddedSample = forrierPad(drumAudio.array[:,0])

# nSamples = 100
# fourier_points = forrierFreqsToHuman(nSamples, np.absolute(fastFourierTransform(paddedSample)), drumAudio.sampleRate)
# plt.xscale("log", base=2)
# plt.xticks(ticks=[20,50,100,200,500,1000,2000,5000,10000,20000], labels=['20','50','100','200','500','1k','2k','5k','10k','20k'])
# plt.plot(humanFrequencySamples(nSamples), fourier_points)
# plt.show()
