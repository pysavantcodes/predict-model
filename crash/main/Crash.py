import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from scipy import stats
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)


def process_data():
    # Get the absolute path to the data file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'data.csv')

    # Load the data
    data = pd.read_csv(data_path)

    # Convert the timestamp to datetime format for easier manipulation
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms')

    # Feature Engineering
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data['Month'] = data['Timestamp'].dt.month

    for lag in range(1, 6):
        data[f'Multiplier_lag_{lag}'] = data['Multiplier'].shift(lag)

    data['Multiplier_roll_mean_5'] = data['Multiplier'].rolling(window=5).mean()
    data['Multiplier_roll_std_5'] = data['Multiplier'].rolling(window=5).std()

    data.dropna(inplace=True)

    X = data.drop(columns=['Multiplier', 'Timestamp'])
    y = data['Multiplier']

    # Log the lengths
    logging.debug(f"Length of X: {len(X)}")
    logging.debug(f"Length of y: {len(y)}")

    return X, y


def run_crash_analysis():
    X, y = process_data()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_

    cv_scores = cross_val_score(best_model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)

    y_pred = best_model.predict(X)

    def certified_fair_crash(crash_points):
        mean_crash = np.mean(crash_points)
        std_dev_crash = np.std(crash_points)
        median_crash = np.median(crash_points)
        skewness = pd.Series(crash_points).skew()
        weighted_prediction = (mean_crash * 0.4 + std_dev_crash * 0.2 +
                               median_crash * 0.3 + skewness * 0.1)
        return round(max(1.0, weighted_prediction), 2)

    smp = certified_fair_crash(y_pred)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    accuracy = r2 * 100

    mean_actual = np.mean(y)
    smp_accuracy = 100 - (abs(smp - mean_actual) / mean_actual * 100)
    smp_deviation = np.mean(abs(y - smp))
    smp_dataset_accuracy = 100 - (smp_deviation / mean_actual * 100)

    result = {
        "Best Parameters": grid_search.best_params_,
        "Prediction List": y_pred,
        "Algorithm Prediction": smp,
        "Root Mean Squared Error": rmse,
        "R^2 Score": r2,
        "Accuracy": accuracy,
        "SMP Accuracy": smp_accuracy,
        "SMP Dataset Accuracy": smp_dataset_accuracy
    }

    return result
