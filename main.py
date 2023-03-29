"""
Creates a series of models, and trains them 
"""
from pathos.multiprocessing import ProcessingPool as Pool
import functools
from matplotlib import pyplot as plt
import numpy as np
from numpy import typing as npt
import tensorflow
from keras import datasets, layers, models, losses
import keras
import audio_files
import audio_processing


SAMPLE_RATE = 44100

def open_drum_dataset(index_file: str, data_folder: str, verbose: int = 5):
    """
    Assumes there index_file is a text document containing a list of filenames of drum
    Samples (one on each line), contained with data_folder
    """
    with open(index_file, "r") as index:
        filenames = index.readlines()

    filenames = [filename.strip("\n") for filename in filenames]
    dataset = []
    for i, filename in enumerate(filenames):
        if verbose != 0 and i%verbose == 0:
            print(filename)
        dataset.append([
            filename,
            audio_files.read_wave_file(data_folder+"/"+filename)
        ])
    return dataset

def create_randomised_datapoint(num_args, drum_machine, drum_sample_frames, sample_rate, i):
    if (i%160) == 0:
          print(i)
    input_ = np.random.uniform(low= 0, high = 1, size = num_args)
    _ , output = drum_machine.eval_array(drum_sample_frames, sample_rate, input_)
    output_spectogram, _  = audio_files.calculate_spectrogram(output, sample_rate, 0.025, 0.1,
                                                              audio_files.FrequencyScale.MEL)
    return (input_, output_spectogram)

def create_randomised_dataset(
        drum_machine: audio_processing.AudioProcessingBlock,
        num_datapoints: int,
        drum_sample_seconds: float,
        ) -> npt.NDArray:
    "Create a randomised dataset of inputs and outputs for a given drum machien"

    inputs_filename  = drum_machine.name.replace(" ", "_") + "_inputs.npy"
    spectrograms_filename = drum_machine.name.replace(" ", "_") + "_spectrograms.npy"
    spectrogram_times_filename = drum_machine.name.replace(" ", "_") + "_spectrogram_times.npy"

    try:
        previous_inputs = np.load(inputs_filename)
    except OSError:
        previous_inputs = None

    try:
        previous_spectrograms = np.load(spectrograms_filename)
    except OSError:
        previous_spectrograms = None
    
    spectrogram_times = None
    try:
        spectrogram_times = np.load(spectrogram_times_filename)
    except OSError:
        spectrogram_times = None
    
    num_precalculated = 0
    if not previous_inputs is None:
        num_precalculated = previous_inputs.shape[0]
        assert previous_inputs.shape[0] == previous_spectrograms.shape[0]

    # If the we already have enough datapoints saved, just use them
    if num_precalculated >= num_datapoints:
        return previous_inputs[:num_datapoints], previous_spectrograms[:num_datapoints], spectrogram_times

    drum_sample_frames = int(drum_sample_seconds*SAMPLE_RATE)
    num_args = len(drum_machine.array_labels())


    calculate_sample_partial = functools.partial(
        create_randomised_datapoint, num_args, drum_machine, drum_sample_frames, SAMPLE_RATE
    )
    with Pool() as pool:
        results = pool.map(calculate_sample_partial,
                            range(num_datapoints - num_precalculated))
        pool.close()
        pool.join()
    
    inputs = np.array([result[0] for result in results])
    spectrograms = np.array([result[1] for result in results])

    if not previous_inputs is None:
        inputs = np.append(previous_inputs, inputs, axis = 0)
    
    if not previous_spectrograms is None:
        spectrograms = np.append(previous_spectrograms, spectrograms, axis = 0)

    if spectrogram_times is None:
        input_ = inputs[0]
        _ , output = drum_machine.eval_array(drum_sample_frames, SAMPLE_RATE, input_)
        _, spectrogram_times  = audio_files.calculate_spectrogram(output, SAMPLE_RATE, 0.1, 0.1,
                                                              audio_files.FrequencyScale.MEL)

    np.save(inputs_filename, inputs)
    np.save(spectrograms_filename, spectrograms)
    np.save(spectrogram_times_filename, spectrogram_times)
    return inputs, spectrograms, spectrogram_times

INPUT_MAPPINGS = {
    "freq" : audio_processing.human_frequency_mapper_mel,
    "amp": lambda x: x,
    "phase": lambda x: x*np.pi*2,
    "vol_control": lambda x: x,
    "threshold": lambda x: 0.6+0.4*x,
    "tension": lambda x: 6*x - 3,
    "power": audio_processing.construct_mapper(0.1,10,
                                               scale=audio_files.FrequencyScale.LOG2.value),
    "time": lambda x: x/2,
    "scale": lambda x: x*2 -1,
    "offset": lambda x: x*2 -1
}

# drum processing chain for kicks

drum_processing_chain_1 = audio_processing.WaveMixGen("generator") # pylint: disable=E1120
drum_processing_chain_1.set_argument("freq", audio_processing.ADSHRT_Envelope())
drum_processing_chain_1.set_argument("amp", audio_processing.ADSHRT_Envelope())
drum_processing_chain_1.set_argument_fixed("phase", 0)
drum_processing_chain_1.set_argument("freq", audio_processing.ADSHRT_Envelope())
drum_processing_chain_1.set_argument("sawToTriangle", audio_processing.ADSHRT_Envelope())
drum_processing_chain_1.set_argument("toSquare", audio_processing.ADSHRT_Envelope())
drum_processing_chain_1.set_argument("toSin", audio_processing.ADSHRT_Envelope())

drum_processing_chain_1.set_mappings_recursive(INPUT_MAPPINGS)


#### General Drum Processing Chain ####


drum_processing_chain_2 = audio_processing.LimiterRescale("Any Drum 2") # pylint: disable=E1120
mix_node_2 = audio_processing.MixNode("mix") # pylint: disable=E1120
drum_processing_chain_2.set_argument("signal", mix_node_2)

deterministic_input_2 = audio_processing.WaveMixGen("generator") # pylint: disable=E1120
deterministic_input_2.set_argument("freq", audio_processing.ADSHRT_Envelope())
amplitude_deterministic_2 = audio_processing.ADSHRT_Envelope()
amplitude_deterministic_2.set_argument_fixed("offset", 0)
amplitude_deterministic_2.set_argument_fixed("scale", 1)
deterministic_input_2.set_argument("amp", amplitude_deterministic_2)
deterministic_input_2.set_argument_fixed("phase", 0)
deterministic_input_2.set_argument("freq", audio_processing.ADSHRT_Envelope())

noise_gen_2 = audio_processing.NoiseGen("noise")
noise_gen_2.set_argument("amp", audio_processing.ADSHRT_Envelope())
amplitude_noise_2 = audio_processing.ADSHRT_Envelope()
amplitude_noise_2.set_argument_fixed("offset", 0)
amplitude_noise_2.set_argument_fixed("scale", 1)
noise_gen_2.set_argument_fixed("amp", amplitude_noise_2)
noise_low_pass = audio_processing.LowPassFilter("low pass") # pylint: disable=E1120
noise_low_pass.set_argument("signal", noise_gen_2)
noise_low_pass.set_argument("cut_off_frequency", audio_processing.ADSHRT_Envelope())
nondeterministic_input_2 = audio_processing.HighPassFilter("high pass") # pylint: disable=E1120
nondeterministic_input_2.set_argument("signal", noise_low_pass)
nondeterministic_input_2.set_argument("cut_off_frequency", audio_processing.ADSHRT_Envelope())

mix_node_2.set_argument("a", deterministic_input_2)
mix_node_2.set_argument("b", nondeterministic_input_2)


drum_processing_chain_2.set_mappings_recursive(INPUT_MAPPINGS)

drum_processing_chain_2_num_inputs = len(drum_processing_chain_2.array_labels())

#### models ####

input_shape = (44, 400, 1)

model_2_CNN = models.Sequential()
model_2_CNN.add(layers.Conv2D(64, (2,3), activation='relu', input_shape=input_shape))
model_2_CNN.add(layers.MaxPooling2D((1, 2)))
model_2_CNN.add(layers.Conv2D(64, (1,3), activation='relu', input_shape=input_shape))
model_2_CNN.add(layers.MaxPooling2D((2, 2)))
model_2_CNN.add(layers.Conv2D(64, (2, 3), activation='relu'))
model_2_CNN.add(layers.Flatten())
model_2_CNN.add(layers.Dense(drum_processing_chain_2_num_inputs*7, activation='relu'))
model_2_CNN.add(layers.Dense(drum_processing_chain_2_num_inputs))

print("model_2_CNN")
print(model_2_CNN.summary())

model_2_CNN.compile(optimizer='adam',
              loss=losses.MeanAbsoluteError(),
              metrics=['accuracy'])



def test_generation():
    "Test both drum machines on random inputs"
    # print(drum_processing_chain_1.array_labels_str())
    inputs, outputs, spectogram_times = create_randomised_dataset(drum_processing_chain_2, 10000, 1)
    print("Done")
    print(inputs)
    print(outputs)
    print("shape outputs", outputs.shape)
    for i in range(2):
        input_ = inputs[i]
        output = outputs[i]
        print()
        print()
        print("output shape", output.shape)
        #print(drum_processing_chain_2.array_labels_and_values_str(input_))
        print()
        audio_files.show_spectrogram(output,spectogram_times, SAMPLE_RATE, scale=audio_files.FrequencyScale.MEL)
      
    # open_drum_dataset("MDLib2.2/filenames.txt", "MDLib2.2/Flat")

def main():
    "main function"
    drum_processing_chain = drum_processing_chain_2
    model = model_2_CNN
    model_name = "model_2_CNN"

    dataset_size = 10000
    inputs, spectrograms, spectogram_times = create_randomised_dataset(drum_processing_chain, dataset_size, 1)
    spectrograms = np.abs(spectrograms)

    train_percent = 70
    train_num = int(dataset_size*train_percent/100)
    test_num = dataset_size - train_num

    inputs_train = inputs[:train_num]
    spectrograms_train = spectrograms[:train_num]

    inputs_test = inputs[train_num:]
    spectrograms_test = spectrograms[train_num:]

    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=model_name+"/cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)
    
    history = model.fit(spectrograms_train, inputs_train, epochs=100, 
                    validation_data=(spectrograms_test, inputs_test), verbose=1,
                    callbacks=[checkpoint_callback])
    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(spectrograms_test,  inputs_test, verbose=2)
    print(test_acc)
    

if __name__ == "__main__":
    main()
