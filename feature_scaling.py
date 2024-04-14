import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Function to read features from the text file

def read_features_from_file(file_path):
    features_dict = {}
    current_audio_file = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('Features for'):
                current_audio_file = line.split('Features for ')[1]
                features_dict[current_audio_file] = {}
            elif line.startswith('MFCCs'):
                feature_name = 'MFCCs'
                feature_shape = line.split(': ')[1]
                feature_shape = tuple(map(int, feature_shape[:-1].split(', ')))
                features_dict[current_audio_file][feature_name] = feature_shape
            elif line.startswith('Pitch'):
                feature_name = 'Pitch'
                feature_shape = line.split(': ')[1]
                feature_shape = tuple(map(int, feature_shape[:-1].split(', ')))
                features_dict[current_audio_file][feature_name] = feature_shape
            elif line.startswith('Intensity'):
                feature_name = 'Intensity'
                feature_shape = line.split(': ')[1]
                feature_shape = tuple(map(int, feature_shape[:-1].split(', ')))
                features_dict[current_audio_file][feature_name] = feature_shape
            elif line.startswith('Spectral Centroid'):
                feature_name = 'Spectral Centroid'
                feature_shape = line.split(': ')[1]
                feature_shape = tuple(map(int, feature_shape[:-1].split(', ')))
                features_dict[current_audio_file][feature_name] = feature_shape
            elif line.startswith('HNR'):
                feature_name = 'HNR'
                feature_shape = line.split(': ')[1]
                feature_shape = tuple(map(int, feature_shape[:-1].split(', ')))
                features_dict[current_audio_file][feature_name] = feature_shape
            elif line.startswith('Sampling Rate'):
                features_dict[current_audio_file]['Sampling Rate'] = int(line.split(': ')[1])
    return features_dict

# Function to perform Min-Max Scaling
def min_max_scaling(features_dict):
    scaler = MinMaxScaler()
    scaled_features = {}
    for audio_file, feature_data in features_dict.items():
        scaled_feature_data = {}
        for feature_name, feature_values in feature_data.items():
            scaled_feature_values = scaler.fit_transform(feature_values.reshape(-1, 1))
            scaled_feature_data[feature_name] = scaled_feature_values
        scaled_features[audio_file] = scaled_feature_data
    return scaled_features

# Function to perform Standardization (Z-score normalization)
def standardization(features_dict):
    scaler = StandardScaler()
    scaled_features = {}
    for audio_file, feature_data in features_dict.items():
        scaled_feature_data = {}
        for feature_name, feature_values in feature_data.items():
            scaled_feature_values = scaler.fit_transform(feature_values.reshape(-1, 1))
            scaled_feature_data[feature_name] = scaled_feature_values
        scaled_features[audio_file] = scaled_feature_data
    return scaled_features

# Example usage
file_path = "feat_extrac_op1.txt"  # Replace with the path to your feat_extrac_op1.txt file

# Read features from the text file
features_dict = read_features_from_file(file_path)

# Perform Min-Max Scaling
min_max_scaled_features = min_max_scaling(features_dict)

# Perform Standardization
standardized_features = standardization(features_dict)

print("Loaded features:")
print(features_dict)
