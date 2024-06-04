import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Step 1: Load and Preprocess the Data
def load_data(file_path):
    # Load data from a CSV file
    data = pd.read_csv(file_path)
    # Assume the last column is the label and others are features
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def preprocess_data(X, y):
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Reshape for CNN input
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

# Step 2: Build and Train the Model
def build_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return history

# Step 3: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))

# Step 4: Make Predictions
def predict(model, X_new, threshold=0.5):
    X_new = X_new.reshape((1, X_new.shape[0], 1))
    probability = model.predict(X_new)[0][0]
    label = int(probability >= threshold)
    return probability, label

# Plot ECG signal
def plot_ecg(signal, label):
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(f"ECG Signal - {'Arrhythmia' if label == 1 else 'Normal'}")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

# Plot heart anatomy with anomaly highlighted
def plot_heart_anatomy(anomaly_type):
    # Load the heart anatomy image
    img = mpimg.imread('heart.jpg')  # Replace with your actual image file
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    # Highlight areas based on the type of arrhythmia
    if anomaly_type == 'Arrhythmia':
        # Example positions, these need to be adjusted based on the actual image
        atria_coords = (606, 495)
        ventricles_coords = (766, 782)
        
        # Highlight atria
        atria_circle = plt.Circle(atria_coords, 50, color='r', alpha=0.5)
        ax.add_patch(atria_circle)
        
        # Highlight ventricles
        ventricles_circle = plt.Circle(ventricles_coords, 50, color='r', alpha=0.5)
        ax.add_patch(ventricles_circle)
        
    plt.title(f"Heart Anatomy - {'Arrhythmia' if anomaly_type == 'Arrhythmia' else 'Normal'}")
    plt.axis('off')
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    file_path = '/Users/auxni/Documents/Science/Programming/ECG/ecg.csv'  # Replace with your actual file path
    X, y = load_data(file_path)
    X, y = preprocess_data(X, y)

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build and train the model
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape)
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Example prediction on new data
    X_new = X_test[0]  # Replace with actual new data
    probability, label = predict(model, X_new)
    print(f"Predicted probability: {probability}")
    print(f"Predicted label: {'Arrhythmia Detected' if label == 1 else 'Normal'}")

    # Plot the ECG signal
    plot_ecg(X_new[:, 0], label)

    # Plot heart anatomy with anomaly highlighted
    plot_heart_anatomy('Arrhythmia' if label == 1 else 'Normal')






