import librosa
import numpy as np
import os

def extract_audio_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Extract pitch
    pitch = librosa.yin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'))
    
    # Extract intensity
    intensity = np.abs(y)
    
    # Extract spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    # Extract harmonic-to-noise ratio (HNR)
    harmonic, percussive = librosa.effects.hpss(y)
    hnr = librosa.effects.harmonic(y)
    
    return mfccs, pitch, intensity, spectral_centroid, hnr, sr

def save_features_to_txt(features_dict, output_folder):
    for audio_file, features in features_dict.items():
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Construct the output file path
        output_file = os.path.join(output_folder, os.path.basename(audio_file) + ".txt")
        
        with open(output_file, 'w') as file:
            file.write(f'Features for {audio_file}\n\n')
            file.write(f'MFCCs: {features["mfccs"].shape}\n')
            file.write(f'Pitch: {features["pitch"].shape}\n')
            file.write(f'Intensity: {features["intensity"].shape}\n')
            file.write(f'Spectral Centroid: {features["spectral_centroid"].shape}\n')
            file.write(f'HNR: {features["hnr"].shape}\n')
            file.write(f'Sampling Rate: {features["sr"]}\n\n')

def process_audio_folder(folder):
    train_data = {}
    test_data = {}
    
    # List all audio files in the folder
    audio_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.ogg'))]
    
    # Separate the first 100 files for training and the next 38 files for testing
    for i, audio_file in enumerate(audio_files):
        mfccs, pitch, intensity, spectral_centroid, hnr, sr = extract_audio_features(audio_file)
        if i < 100:
            train_data[audio_file] = {"mfccs": mfccs, "pitch": pitch, "intensity": intensity, 
                                      "spectral_centroid": spectral_centroid, "hnr": hnr, "sr": sr}
        else:
            test_data[audio_file] = {"mfccs": mfccs, "pitch": pitch, "intensity": intensity, 
                                     "spectral_centroid": spectral_centroid, "hnr": hnr, "sr": sr}
    
    return train_data, test_data

# Example usage
output_folder = "output1"
train_output_folder = "train_data"
test_output_folder = "test_data"

# Extract features from all audio files in the output1 folder
train_data, test_data = process_audio_folder(output_folder)

# Save features to separate text files for training and testing
save_features_to_txt(train_data, train_output_folder)
save_features_to_txt(test_data, test_output_folder)
