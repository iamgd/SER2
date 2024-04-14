import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the extracted MFCC data along with the corresponding emotion labels
def load_data(excel_file):
    df = pd.read_excel(excel_file)
    X = df.drop('Emotion', axis=1)  # Features
    y = df['Emotion']  # Labels
    return X, y

# Step 2: Split the data into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Step 3: Feature Scaling (Optional)
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Step 4: Train the Classifier
def train_classifier(X_train, y_train):
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(X_train, y_train)
    return classifier

# Step 5: Evaluate the Classifier
def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Step 6: Predict Emotions (Optional)
def predict_emotions(classifier, new_data):
    return classifier.predict(new_data)

# Main function
if __name__ == "__main__":
    # Step 1: Load data
    excel_file = "extracted_features.xlsx"
    X, y = load_data(excel_file)
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 3: Feature scaling (optional)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Step 4: Train classifier
    classifier = train_classifier(X_train_scaled, y_train)
    
    # Step 5: Evaluate classifier
    accuracy, report = evaluate_classifier(classifier, X_test_scaled, y_test)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    
    # Step 6: Predict emotions (optional)
    predicted_emotions = predict_emotions(classifier, new_data)
    print("Predicted emotions:", predicted_emotions)
