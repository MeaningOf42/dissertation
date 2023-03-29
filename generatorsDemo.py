from audio_processing import *

def main():
    inputArray = [50.1,1.1,0.1]
    
    gen = SinGen("Sin")
    audio=gen.arrayToAudio(44000, 2, inputArray)
    audio.plot()
    audio.showSpectrogram()

    gen = SquareGen("Square")
    audio=gen.arrayToAudio(44000, 2, inputArray)
    audio.plot()
    audio.showSpectrogram()

    gen = SawGen("Saw")
    audio=gen.arrayToAudio(44000, 2, inputArray)
    audio.plot()
    audio.showSpectrogram()

    gen = TriangleGen("Triangle")
    audio = gen.arrayToAudio(44000, 2, inputArray)
    audio.plot()
    audio.showSpectrogram()

    inputArray = [1]
    gen = NoiseGen("Noise")
    audio = gen.arrayToAudio(44000, 2, inputArray)
    audio.plot()
    audio.showSpectrogram()

    
    

if __name__ == "__main__":
    main()
