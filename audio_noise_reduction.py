from pydub import AudioSegment
import numpy as np
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff_freq, fs, order=3):
    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y.astype(np.int16)

def noise_reduction_with_lowpass(audio_path, output_path, cutoff_freq=2000, order=3):
    # Load audio file
    audio = AudioSegment.from_file(audio_path)

    # Convert audio to numpy array
    audio_data = np.array(audio.get_array_of_samples())

    # Apply low-pass filter for noise reduction
    filtered_audio_data = apply_lowpass_filter(audio_data, cutoff_freq, audio.frame_rate, order)

    # Convert back to audio
    cleaned_audio = AudioSegment(
        filtered_audio_data.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )

    # Save cleaned audio
    cleaned_audio.export(output_path, format="wav")
    print(f"Cleaned audio saved to {output_path}")

# Example usage
if __name__ == "__main__":
    audio_path = "audio_file_with_noise.wav"
    output_path = "cleaned_audio.wav"
    
    noise_reduction_with_lowpass(audio_path, output_path)
