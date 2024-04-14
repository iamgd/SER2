import librosa
import numpy as np

def extract_prosodic_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file)

    # Extract pitch (fundamental frequency) using librosa's pitch detection
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

    # Compute mean pitch
    mean_pitch = np.nanmean(pitches)

    # Extract intensity (root mean square energy)
    rms = librosa.feature.rms(y=y)

    # Compute mean intensity
    mean_intensity = np.mean(rms)

    # Extract duration (total duration of the audio file)
    duration = librosa.get_duration(y=y, sr=sr)

    return mean_pitch, mean_intensity, duration

# Example usage
if __name__ == "__main__":
    audio_file = "output/cleaned_03-01-01-01-01-01-01.wav"
    pitch, intensity, duration = extract_prosodic_features(audio_file)
    print("Mean Pitch:", pitch)
    print("Mean Intensity:", intensity)
    print("Duration:", duration)
