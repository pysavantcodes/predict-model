import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow GPU-related messages

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
import io
import requests

def predict_next_multiplier():
    # Load the second dataset
    url = "http://nmehwp-ip-105-113-108-56.tunnelmole.net/data.csv"
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))

    # Select features and target
    features = df[['Total bets', 'Timestamp']]
    target = df['Multiplier']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the neural network model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer for regression (predicting a single continuous value)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_split=0.2)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss: {test_loss}")

    # Predict the multiplier for the topmost entry in the DataFrame
    input_data = X_test_scaled[:1]
    predicted_multiplier = model.predict(input_data)[0][0]

    print(f"Predicted Multiplier: {predicted_multiplier}")
    return predicted_multiplier

# predict_next_multiplier()
