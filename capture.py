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
        audio= np.pad(audio,(0, target_len - len(audio)))
    mfcc = librosa.feature.mfcc(y = audio, sr= RATE, n_mfcc= N_MFCC)
    return np.mean(mfcc.T, axis = 0)

def extract_from_array(audio_array):
    """
   Extact the MFCC froma a raw float32 array of audio data. 
   Used for real-time feature extraction during inference.
   audio_array is a float32 numpy array with values in the range [-1.0, 1.0].
    """
    mfcc = librosa.feature.mfcc(y=audio_array, sr=RATE, n_mfcc = N_MFCC)
    return np.mean(mfcc.T, axis=0)
