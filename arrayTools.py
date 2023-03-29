# frequent operations that make np easier to use
import numpy as np

def padTowidth(array, width):
    return np.pad(array, (0,width-len(array)))

def padToLenOfArray(array, arrayToPadTo):
    return np.pad(array, (0,len(arrayToPadTo)-len(array)))
