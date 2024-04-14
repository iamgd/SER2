import os
import pandas as pd
import numpy as np  # Add this import statement
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_combined_mfcc(excel_file):
    # Load the extracted features from the Excel file
    df = pd.read_excel(excel_file, index_col=0)

    # Initialize an empty list to store all MFCC data
    all_mfccs = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the MFCCs data from the row and append it to the list
        mfccs = row.values
        all_mfccs.append(mfccs)

    # Convert the list of MFCC data to a numpy array
    all_mfccs = np.array(all_mfccs)

    # Plot all MFCCs as a heatmap
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(all_mfccs.T, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Combined MFCCs for all Audio Files')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Specify the path to the Excel file containing the extracted features
    extracted_features_excel_file = "extracted_features.xlsx"
    
    # Plot combined MFCCs for all files
    plot_combined_mfcc(extracted_features_excel_file)
