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

def save_features_to_txt(features_dict, output_file):
    with open(output_file, 'w') as file:
        for audio_file, features in features_dict.items():
            file.write(f'Features for {audio_file}\n\n')
            file.write(f'MFCCs: {features["mfccs"].shape}\n')
            file.write(f'Pitch: {features["pitch"].shape}\n')
            file.write(f'Intensity: {features["intensity"].shape}\n')
            file.write(f'Spectral Centroid: {features["spectral_centroid"].shape}\n')
            file.write(f'HNR: {features["hnr"].shape}\n')
            file.write(f'Sampling Rate: {features["sr"]}\n\n')

def process_audio_folder(folder):
    features_dict = {}
    # List all audio files in the folder
    audio_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.ogg'))]
    if audio_files:
        audio_file = audio_files[0]  # Select the first file
        # Extract features for the selected audio file
        mfccs, pitch, intensity, spectral_centroid, hnr, sr = extract_audio_features(audio_file)
        features_dict[audio_file] = {"mfccs": mfccs, "pitch": pitch, "intensity": intensity, 
                                      "spectral_centroid": spectral_centroid, "hnr": hnr, "sr": sr}
    return features_dict

# Example usage
output_folder = "output1"
output_file = "feat_extrac_op_1.txt"

# Extract features from the first audio file in the output1 folder
features_dict = process_audio_folder(output_folder)

# Save features to a single text file
save_features_to_txt(features_dict, output_file)
