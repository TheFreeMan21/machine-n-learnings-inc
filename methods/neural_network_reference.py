'''Feed-Forward Neural Network regressor - reference implementation'''

# ...
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util.filter_data import filtering

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

df = filtering('src/claims_train.csv', alpha=2, gamma=0.1, train=True)

Y = df['Risk'].values.astype(np.float32).reshape(-1, 1)
df_clean = df.drop(columns=["IDpol", "ClaimNb", "Exposure","Region",'Risk'])
numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df_clean.select_dtypes(include=["object", "category"]).columns

X_cat = pd.get_dummies(df_clean[categorical_cols], drop_first=True)
df_clean = df_clean.drop(columns=categorical_cols)
X_final = np.hstack([df_clean, X_cat.values])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final).astype(np.float32)

# Flatten Y
y_flat = Y.ravel()

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# 80/20 train–test split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# Define neural network
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=50,          # number of epochs
    batch_size=100,       # mini-batch size
    random_state=42,
    verbose=True          # prints loss per iteration
)

# Train
mlp.fit(X_train, y_train)

# Predict
y_train_pred = mlp.predict(X_train)
y_val_pred   = mlp.predict(X_val)

# Compute MSE
mse_train_mlp = mean_squared_error(y_train, y_train_pred)
mse_val_mlp   = mean_squared_error(y_val,   y_val_pred)

print("MLPRegressor train MSE:", mse_train_mlp)
print("MLPRegressor val   MSE:", mse_val_mlp)