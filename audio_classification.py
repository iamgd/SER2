import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def load_features_from_txt(folder):
    features_dict = {}
    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder, file_name)
            features = {}
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith('MFCCs'):
                        feature_name = 'MFCCs'
                        feature_shape = line.split(': ')[1]
                        feature_shape = tuple(map(int, feature_shape[1:-1].split(', ')))
                        features[feature_name] = feature_shape
            features_dict[file_name] = features
    return features_dict

def load_labels_from_folder(folder):
    labels = {}
    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            label = file_name.split("_")[0]  # Extract label from file name
            labels[file_name] = label
    return labels

# Load features and labels for training and testing
train_features = load_features_from_txt("train_data")
train_labels = load_labels_from_folder("train_data")
test_features = load_features_from_txt("test_data")
test_labels = load_labels_from_folder("test_data")

# Convert features to arrays
train_data = np.array([np.zeros(features["MFCCs"]) for features in train_features.values()])
test_data = np.array([np.zeros(features["MFCCs"]) for features in test_features.values()])

# Convert labels to arrays
train_labels = np.array(list(train_labels.values()))
test_labels = np.array(list(test_labels.values()))

# Train SVM classifier
clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
clf.fit(train_data, train_labels)

# Make predictions on the test set
predictions = clf.predict(test_data)

# Evaluate performance
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)
