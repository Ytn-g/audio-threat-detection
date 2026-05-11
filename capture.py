import numpy as np
import librosa
RATE = 16000
N_MFCC = 40
DURATION = 1.0

def extract_from_file(filepath):
    """
    Load a .wav file and return a (40,) MFCC feature vector.
    Used for batch feature extraction during training.
    """
    audio, sr= librosa.load(filepath, sr=RATE, duration= DURATION)
    target_len= int (RATE * DURATION)
    if len(audio)< target_len:
        audio= np.pad