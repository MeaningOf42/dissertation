from collections.abc import Iterable
import inspect
import numpy as np
import audio_files
import arrayTools
from audio_files import FrequencyScale
import copy

class TimeValsDefault:
    """
    Empty class to signify when an argument represents time
    """


TIME_VALS_DEFAULT = TimeValsDefault()


class AudioProcessingBlock():
    """
    Class that represents a node in a dsp chain.
    """
    # pylint: disable=C3001
    def __init__(self, non_array_inputs=None, constant_inputs=None, mappings=None):
        self.func = lambda signal: signal
        self.name = "Straight Wire"
        self.inputs = {"signal": 0}

        self.input_map_names = {}
        # input_map_names defines how an input should be scaled from a value between 0 and 1
        if mappings is None:
            self.mappings = {}
        else:
            self.mappings = mappings

        # NoneArrayInputs are inputs that can't be changed by array
        if non_array_inputs is None:
            self.non_array_inputs = []
        else:
            self.non_array_inputs = non_array_inputs

        self.non_array_inputs += [name for name,
                                  value in self.inputs.items() if value is TIME_VALS_DEFAULT]
        self.non_array_inputs = list(set(self.non_array_inputs))

        # constantInputs are inputs that can not change with time, such as attack time

        if constant_inputs is None:
            self.constant_inputs = []
        else:
            self.constant_inputs = constant_inputs

    def __parse_arg(self, value, name, n_frames, sample_rate):
        """
        Take in any argument and return an array that represents it
        """

        if name in self.constant_inputs:
            assert not (
                isinstance(value, Iterable) or issubclass(
                    type(value), AudioProcessingBlock)
            ), "Parameter must be single value"
            return value

        if isinstance(value, np.ndarray):
            if len(value) < n_frames:
                return np.pad(value.flatten, n_frames-len(value))
            return value.flatten()[:n_frames]

        if value is inspect.Parameter.empty:
            return np.zeros(n_frames)

        if isinstance(value, TimeValsDefault):
            return np.arange(n_frames)/sample_rate

        if issubclass(type(value), AudioProcessingBlock):
            return value.eval(n_frames, sample_rate)

        return np.full(n_frames, value)

    def eval(self, n_frames, sample_rate):
        """
        Evalute based on the currently set inputs, what the output will be as an array
        """
        new_vals = {name: self.__parse_arg(
            value, name, n_frames, sample_rate) for name, value in self.inputs.items()}

        return self.func(**new_vals)

    def eval_array(self, n_frames, sample_rate, array):
        """
        Evalute based on the currently set inputs, and an input array what the output
        will be as an array.
        """
        new_vals = {}
        for name, value in self.inputs.items():

            # works out how to map the inputs
            if name in self.input_map_names and \
                    self.input_map_names[name] in self.mappings:
                scale = self.mappings[self.input_map_names[name]]
            else:
                scale = lambda n: n

            if name in self.non_array_inputs:
                if isinstance(value, np.ndarray):
                    new_vals[name] = self.__parse_arg(
                        value, name, n_frames, sample_rate)
                else:
                    new_vals[name] = self.__parse_arg(
                        value, name, n_frames, sample_rate)

            elif issubclass(type(value), AudioProcessingBlock):
                array, new_vals[name] = value.eval_array(n_frames, sample_rate, array)
                new_vals[name] = scale(new_vals[name])

            else:
                new_vals[name], array = self.__parse_arg(
                    scale(array[0]), name, n_frames, sample_rate), array[1:]

        return (array, self.func(**new_vals))

    def to_audio(self, sample_rate, time):
        return audio_files.DiscreteAudio(self.eval(time*sample_rate, sample_rate), sample_rate)

    def array_to_audio(self, sample_rate, time, array):
        return audio_files.DiscreteAudio(
            self.eval_array(int(time*sample_rate), sample_rate, array)[1],
            sample_rate
        )

    def set_argument(self, name, value):
        assert name in self.inputs, "Can only set arguments the function expects."
        self.inputs[name] = value
    
    def set_argument_fixed(self, name, value):
        "adds an input that the input array does not set"

        assert name in self.inputs, "Can only set arguments the function expects."
        self.inputs[name] = value
        self.non_array_inputs.append(name)
    
    def set_mappings(self, mappings):
        self.mappings = mappings
    
    def set_mappings_recursive(self, mappings):
        self.mappings = mappings
        for _, value in self.inputs.items():
            if issubclass(type(value), AudioProcessingBlock):
                value.set_mappings_recursive(mappings)

    def array_labels(self):
        formated_args = [
            self.name + ": "+a.ljust(max(map(len, self.inputs.keys()))+1, " ") for a in self.inputs]
        array_labels = []

        for name, value in self.inputs.items():
            formated_arg = formated_args.pop(0)
            if not name in self.non_array_inputs:
                if issubclass(type(value), AudioProcessingBlock):
                    value: AudioProcessingBlock
                    array_labels += [formated_arg + "-> " +
                                    label for label in value.array_labels()]

                else:
                    array_labels.append(formated_arg)

        return array_labels

    def array_labels_str(self):
        return "\n".join(self.array_labels())

    def array_labels_and_values(self, array):

        labels = self.array_labels()
        assert len(labels) == len(
            array), "Array must be of length " + str(len(labels)) + "."
        return [label.ljust(max(map(len, labels))+1, " ") + ": "+str(value) for label, value in zip(labels, list(array))]

    def array_labels_and_values_str(self, array):
        return "\n".join(self.array_labels_and_values(array))


def NodeClass(*args, constantInputs=None, input_map_names=None):
    if len(args) == 0:
        def func_wrapper(func):
            class NodeWrapper(AudioProcessingBlock):
                def __init__(self, name=None, nonArrayInputs=None):
                    super().__init__(non_array_inputs=nonArrayInputs, constant_inputs=constantInputs)

                    if name is None:
                        self.name = func.__name__
                    else:
                        self.name = func.__name__ + " " + name

                    self.inputs = {name: param.default
                                   for name, param in inspect.signature(func).parameters.items()
                                   }
                    self.func = func

                    self.non_array_inputs += [name for name,
                                              value in self.inputs.items() if value is TIME_VALS_DEFAULT]
                    self.non_array_inputs = list(set(self.non_array_inputs))

                    self.input_map_names = input_map_names
                    if input_map_names is None:
                        self.input_map_names.input_map_names = {}

            return NodeWrapper

        return func_wrapper

    assert len(args) == 1, "wrapper can only take in one function"

    func = args[0]

    class NodeWrapper(AudioProcessingBlock):
        def __init__(self, name=None, nonArrayInputs=None):
            super().__init__(non_array_inputs=nonArrayInputs)

            if name is None:
                self.name = func.__name__
            else:
                self.name = func.__name__ + " " + name

            self.inputs = {name: param.default
                           for name, param in inspect.signature(func).parameters.items()
                           }
            self.func = func

            self.non_array_inputs += [name for name,
                                      value in self.inputs.items() if value is TIME_VALS_DEFAULT]
            self.non_array_inputs = list(set(self.non_array_inputs))

    return NodeWrapper

def construct_mapper(min_= 0, max_=1, scale=None):
    """
    Creates a function that maps values between 0 and 1 to min and max using the scale
    """
    if scale is None:
        scale = {"transform": lambda f: f, "inverse": lambda scale: scale}

    scaled_min = scale["transform"](min_)
    scaled_size = scale["transform"](max_) - scaled_min


    def mapper(unmapped_input):
        scaled_input = scaled_min + scaled_size*unmapped_input
        return scale["inverse"](scaled_input)

    return mapper

human_frequency_mapper_linear = construct_mapper(
        min_ = audio_files.MIN_HUMAN_FREQ,
        max_ = audio_files.MAX_HUMAN_FREQ,
    )

human_frequency_mapper_mel = construct_mapper(
        min_ = audio_files.MIN_HUMAN_FREQ,
        max_ = audio_files.MAX_HUMAN_FREQ,
        scale = FrequencyScale.MEL.value
    )

human_frequency_mapper_log2 = construct_mapper(
        min_ = audio_files.MIN_HUMAN_FREQ,
        max_ = audio_files.MAX_HUMAN_FREQ,
        scale = FrequencyScale.LOG2.value
    )




# pylint: disable=C0103
@NodeClass
def AddNode(a, b):
    "Adds two signals together"
    return (a + b)/2


@NodeClass(input_map_names={"freq":"freq", "amp":"amp", "phase":"phase"})
def SinGen(time=TIME_VALS_DEFAULT, freq=1, amp=1, phase=0):
    "Generates a sin wave"
    return np.sin(time*freq*2*np.pi + phase) * amp


@NodeClass(input_map_names={"freq":"freq", "amp":"amp", "phase":"phase"})
def SquareGen(time=TIME_VALS_DEFAULT, freq=1, amp=1, phase=0):
    "Generates a square wave"
    phase += 1/2/freq
    return np.sign((time+phase) % (1/freq)-1/2/freq)*amp


@NodeClass(input_map_names={"freq":"freq", "amp":"amp", "phase":"phase"})
def SawGen(time=TIME_VALS_DEFAULT, freq=1, amp=1, phase=0):
    "Generates a saw wave"
    phase += 1/2/freq
    return ((time+phase) % (1/freq)-1/2/freq)*2*freq*amp


@NodeClass(input_map_names={"freq":"freq", "amp":"amp", "phase":"phase"})
def TriangleGen(time=TIME_VALS_DEFAULT, freq=1, amp=1, phase=0):
    "Generates a triangle wave"
    phase -= 1/4/freq
    return (np.abs(((time+phase) % (1/freq)-1/2/freq))*4*freq-1)*amp


@NodeClass(input_map_names={"freq":"freq", "amp":"amp", "phase":"phase"})
def NoiseGen(time=TIME_VALS_DEFAULT, amp=1):
    "Generates white noise"
    return np.random.uniform(low=-amp, high=amp, size=len(time))

@NodeClass(input_map_names={"freq":"freq", "amp":"amp", "phase":"phase", "power":"power"})
def SinPowGen(time=TIME_VALS_DEFAULT, freq=1, amp=1, phase=0, power=1):
    no_power = np.sin(time*freq*2*np.pi + phase)
    return np.abs(no_power)**power * np.sign(no_power) * amp


@NodeClass(input_map_names={"freq":"freq", "amp":"amp", "phase":"phase"})
def SinToSawToothGen(time=TIME_VALS_DEFAULT, freq=1, amp=1, phase=0, saw=1):
    return np.arctan(saw*np.sin(time*freq*2*np.pi + phase) / (1-saw*np.cos(time*freq*2*np.pi + phase)))*amp/saw


@NodeClass(input_map_names={"freq":"freq", "amp":"amp", "phase":"phase", "power":"power"})
def SinToSawToothPowGen(time=TIME_VALS_DEFAULT, freq=1, amp=1, phase=0, saw=1, power=1):
    noPower = np.arctan(saw*np.sin(time*freq*2*np.pi + phase) /
                        (1-saw*np.cos(time*freq*2*np.pi + phase)))/saw
    return np.abs(noPower)**power * np.sign(noPower) * amp


def mix_2_values(a, b, mix):
    return a*(1-mix)+b*mix

@NodeClass(input_map_names={"mix":"vol_control"})
def MixNode(a, b, mix):
    "Mixes two Signals"
    return mix_2_values(a, b, mix)


@NodeClass(input_map_names={"freq":"freq", "amp":"amp", "phase":"phase",
                            "sawToTriangle":"vol_control", "toSquare":"vol_control",
                            "toSin":"vol_control"})
def WaveMixGen(time=TIME_VALS_DEFAULT, freq=1, amp=1, phase=0, sawToTriangle=1, toSquare=1, toSin=1):

    sin = np.sin(time*freq*2*np.pi + phase)
    saw = ((time+phase+1/2/freq) % (1/freq)-1/2/freq)*2*freq
    triangle = (np.abs(((time+phase-1/4/freq) % (1/freq)-1/2/freq))*4*freq-1)
    square = np.sign((time+phase+1/2/freq) % (1/freq)-1/2/freq)

    noScale = mix_2_values(mix_2_values(mix_2_values(
        saw, triangle, sawToTriangle), square, toSquare), sin, toSin)

    return noScale*amp/max(np.abs(noScale))


@NodeClass(constantInputs=["attack", "release"], input_map_names={"attack":"time", "release":"time"})
def AR_Envelope(time=TIME_VALS_DEFAULT, attack=0.5, release=0.5):
    ar_curves  = np.array([
        time/attack,
        1-(time-attack)/release
    ])
    zeros = np.zeros(time.shape)

    return np.maximum(zeros, np.min(ar_curves, axis=0))

@NodeClass(constantInputs=["attack", "hold", "release"],
           input_map_names={"attack":"time", "decay":"time", "hold":"time"})
def AHR_Envelope(time=TIME_VALS_DEFAULT, attack=0.5, hold=0.5, release=0.5):
    ahr_curves  = np.array([
        time/attack,
        np.ones(time.shape),
        1-(time-attack-hold)/release
    ])
    zeros = np.zeros(time.shape)

    return np.maximum(zeros, np.min(ahr_curves, axis=0))

@NodeClass(constantInputs=["attack", "hold", "decay", "sustain", "release"],
           input_map_names={"attack":"time", "decay":"time", "sustain":"amp","hold":"time", "release":"time"})
def ADSHR_Envelope(time=TIME_VALS_DEFAULT, attack=0.5, decay=0.1,
                   sustain=0.9, hold=0.5, release=0.5):

    decay_gradient = (sustain-1)/decay

    decay_sustain_curve = np.maximum(1+(time-attack)*decay_gradient, np.ones(time.shape)*sustain)
    ahsr_curves  = np.array([
        time/attack,
        decay_sustain_curve,
        sustain-(time-attack-decay-hold)/release
    ])
    zeros = np.zeros(time.shape)

    #return decay_sustain_curve
    return np.maximum(zeros, np.min(ahsr_curves, axis=0))


def tension_curve_unscalled(T):
    """
    Creates a curve between the points (0,1) and (1,0) that is more tight as T increases
    """
    if T < 0:
        T = 10**-T
        return np.vectorize(lambda x: 1 + x/T - (T**-x))

    T = 10**T
    return np.vectorize(lambda x: T**(x-1) - (1-x)/T)

def tension_curve(x1, y1, x2, y2, T):
    """
    Return a curve of tension T between the two points that goes through
    the points (x1,y1) and (x2,y2)
    """
    curve_unscalled = tension_curve_unscalled(T)
    y_scale = y2-y1
    x_scale = x2-x1

    def curve(x):
        return y1 + y_scale*curve_unscalled((x-x1)/x_scale)
    return curve


@NodeClass(constantInputs=["attack", "attack_tension", "hold",
                           "release", "release_tension"],

           input_map_names={"attack":"time", "attack_tension":"tension",
                            "hold":"time",
                            "release":"time", "release_tension":"tension"}
                            )
def AHRT_Envelope(time=TIME_VALS_DEFAULT, attack=0.5, attack_tension=3, hold=1,
                  release=0.5, release_tension = -5):
    
    attack_curve = tension_curve(0,0, attack,1, attack_tension)(time)
    release_curve = tension_curve(attack+hold,1, attack+hold+release, 0, release_tension)(time)
    ahr_curves  = np.array([
        attack_curve,
        np.ones(time.shape),
        release_curve
    ])
    zeros = np.zeros(time.shape)

    return np.maximum(zeros, np.min(ahr_curves, axis=0))

@NodeClass(constantInputs=["attack", "attack_tension", "decay", "decay_tension", "sustain"
                           "hold", "release", "release_tension", "scale", "offset"],

           input_map_names={"attack":"time", "attack_tension":"tension",
                            "decay":"time", "decay_tension":"tension",
                            "sustain":"amp","hold":"time",
                            "release":"time", "release_tension":"tension",
                            "scale":"scale", "offset":"offset"}
                            )
def ADSHRT_Envelope(time=TIME_VALS_DEFAULT, attack=0.5, attack_tension=3, decay = 0.2,
                    decay_tension = 3, sustain = 0.7, hold=1,release=0.5, release_tension = -5,
                    scale = 1, offset = 0):

    attack_curve = tension_curve(0,0, attack,1, attack_tension)(time)
    decay_curve = tension_curve(attack,1, attack+decay,sustain, decay_tension)(time)
    
    release_curve = tension_curve(attack+decay+hold, sustain,
                                  attack+decay+hold+release, 0,
                                  release_tension
                                  )(time)
    sustain_release_curve = np.minimum(release_curve, np.ones(time.shape)*sustain)
    decay_sustain_release_curve = np.maximum(decay_curve, sustain_release_curve)
    ahr_curves  = np.array([
        attack_curve,
        decay_sustain_release_curve
    ])
    zeros = np.zeros(time.shape)

    # return release_curve

    return np.maximum(zeros, np.min(ahr_curves, axis=0))

@np.vectorize
def distortion_response(amp, pre, threshold, tension, linear_in, post, mix):
    "distortion response curve"
    dry = amp
    amp *= pre
    linear_out = linear_in + 0.4*(1-threshold)
    linear_amp = amp*linear_out/linear_in
    distorted_amp = tension_curve(linear_in,linear_out, 1/pre,threshold, tension)(amp)
    distorted_amp = max(linear_out, distorted_amp)
    wet = min(linear_amp, distorted_amp, threshold)*post
    mixed = dry*(1-mix)+wet*mix
    return mixed

@NodeClass(input_map_names={"pre":"vol_control", "linear_in":"linear_in",
                            "threshold":"threshold", "tension":"tension",
                            "post":"vol_control","mix":"vol_control"}
                            )
def Distortion(signal, pre, linear_in, threshold, tension, post, mix):
    "Distortion"
    return np.sign(signal)*distortion_response(np.abs(signal), pre, threshold, tension, linear_in, post, mix)

@NodeClass
def Limiter(signal, limit):
    "Limits maximum amplitude to limit"
    return np.maximum(np.minimum(signal, limit), -limit)

@NodeClass
def LimiterRescale(signal, limit):
    "Limits maximum amplitude to limit, then rescales so the max amplitude is 1"
    return np.maximum(np.minimum(signal, limit), -limit)/limit

# All credit to: https://thewolfsound.com/allpass-based-lowpass-and-highpass-filters/
def all_pass_filter(signal, break_frequency, sample_rate):
    "first order all pass filter"
    # simplification of (tan - 1) / (tan + 1)
    a1_coeffients = np.tan(np.pi*(break_frequency / sample_rate - 1/4))
    #
    # pretty much exactly as found in article
    buffer = 0
    output = np.zeros(signal.size)
    for i in range(signal.size):
        output[i] = a1_coeffients[i] * signal[i] + buffer
        buffer = signal[i] - a1_coeffients[i] * output[i]

    return output

# All credit to: https://thewolfsound.com/allpass-based-lowpass-and-highpass-filters/
@NodeClass(input_map_names={"cut_off_frequency":"freq"})
def HighPassFilter(signal, cut_off_frequency, time=TIME_VALS_DEFAULT):
    "High pass filter"
    sample_rate = int(1/time[1])
    return (signal-all_pass_filter(signal, cut_off_frequency, sample_rate))*0.5

# All credit to: https://thewolfsound.com/allpass-based-lowpass-and-highpass-filters/
@NodeClass(input_map_names={"cut_off_frequency":"freq"})
def LowPassFilter(signal, cut_off_frequency, time=TIME_VALS_DEFAULT):
    "low pass filter"
    sample_rate = int(1/time[1])
    return (signal + all_pass_filter(signal, cut_off_frequency, sample_rate))*0.5


# pylint: enable=C0103


def main():
    """
    sin_gen = SinGen("main")
    sin_gen.set_argument("amp", SinGen("vol mod"))

    freq_control = AddNode("freq mod")
    freq_control.set_argument("a", SinGen("freq osc"))
    sin_gen.set_argument("freq", freq_control)

    # print(sinGen.eval(10, 10))
    # print(sinGen.evalArray(10, 10, [1,1,1,1,1]))
    print("layout of audio processing chain: ")
    print(sin_gen.array_labels_str())

    print()
    print("playing 10 seconds of audio with the following input: ")
    array = [10, 100, 0, 200, 1, 1, 1, 1]
    array = np.random.random(len(array))*100
    array[5] = 1
    print(sin_gen.array_labels_and_values_str(array))
    sin_audio = sin_gen.array_to_audio(44000, 2, array)
    sin_audio.play()
    sin_audio.write("NoiseHatNew.wav")

    sin_audio.plot()
    sin_audio.show_spectrogram()

    noise_gen = NoiseGen("noise generator")
    noise_gen.set_argument("amp", AR_Envelope())
    print(noise_gen.array_labels_str())
    noise_audio = noise_gen.array_to_audio(44000, 2, [1, 0])
    noise_audio.play()
    noise_audio.plot()
    noise_audio.show_spectrogram()
    noise_audio.write("NoiseHatNew.wav")

    sinPowGen = WaveMixGen("main")
    sin_pow_audio = sinPowGen.array_to_audio(44000, 2, np.concatenate((
                                                        np.array([100,0.05,]),
                                                        np.random.random(3),
                                                        np.array([0.5])
                                                        ))
                                                    )
    sin_pow_audio.play()
    sin_pow_audio.write("sinAudio")

    sin_pow_audio.plot()


    sin_gen_clean = SinGen("main")
    sin_gen_clean.set_argument_fixed("freq", 240)
    sin_gen_clean.set_argument_fixed("amp", 0.8)
    sin_gen_clean.set_argument_fixed("phase", 0)
    # pre, linear_in, threshold, tension, post, mix
    distortion_test = Distortion("main") # pylint: disable=E1120
    distortion_test.set_argument("signal", sin_gen_clean)
    inputs = np.array([0, 0.1, -0.4, 6, 0, 0])*np.random.uniform(0,1,6) + np.array([1.2,0.05,1,-3,1,1])
    print(distortion_test.array_labels_str())
    print(distortion_test.array_labels_and_values_str(inputs))
    
    
    distortion_test_audio = distortion_test.array_to_audio(44100, 2, inputs)
    distortion_test_audio.play()

    distortion_test_audio.plot()
"""
    noise_gen_2 = NoiseGen("noise")
    noise_gen_2.set_argument_fixed("amp", 1)
    noise_gen_2_audio = noise_gen_2.to_audio(44100,2)
    noise_gen_2_audio.play()
    noise_gen_2_audio.plot()
    filter_sweep = HighPassFilter("filter sweep") # pylint: disable=E1120
    filter_sweep.set_mappings({"freq":human_frequency_mapper_mel})
    filter_sweep.set_argument("signal", noise_gen_2)
    cutoff_envolope = ADSHR_Envelope("Test Envelope")
    _ , cutoff_frequency = cutoff_envolope.eval_array(44100*8,
                                44100, np.array([8+0.01,0,0,0,0]))
    print("cut off :  ", cutoff_frequency)
    filter_sweep.set_argument("cut_off_frequency", cutoff_envolope)
    print(filter_sweep.array_labels())
    print(human_frequency_mapper_mel(0))
    filter_sweep_audio = filter_sweep.array_to_audio(44100, 8, np.array([8,1,1,1,1,1]))
    filter_sweep_audio.play()
    filter_sweep_audio.plot()




if __name__ == "__main__":
    main()
