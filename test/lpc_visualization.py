import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_lpc(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract LPC coefficients
    lpc = librosa.lpc(y, order=12)  # Using 12 coefficients
    
    # Plot LPC coefficients as a heatmap
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(lpc.reshape(1, -1), x_axis='time')
    plt.colorbar()
    plt.title('LPC')
    plt.xlabel('Time (s)')
    plt.ylabel('LPC Coefficients')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Choose one of the cleaned audio files
    audio_path = "output/cleaned_03-01-01-01-01-01-01.wav"
    
    # Plot LPC coefficients
    plot_lpc(audio_path)
