#Importing the filtering function from util folder
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util.filter_data import filtering

# Essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

df = filtering("./src/claims_train.csv")

X = df.drop(columns=["ClaimNb", "Exposure", "Risk"])
y = df["Risk"]

# Identify column types
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numeric_cols = X.select_dtypes(include=["number"]).columns

# Preprocessing pipeline: OneHot + Scaling
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit and transform training data, transform validation data
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# param_grid = {
#     "n_estimators": [100 ,200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500],
#     "max_depth": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#     "min_samples_split": [5, 10, 15, 20, 50, 100, 150, 200, 250, 300],
#     "min_samples_leaf": [1, 2, 4, 6, 8, 10, 50, 100, 150, 200]
# }

# rf = RandomForestRegressor(random_state=42)
# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv=5,
#     scoring="r2",
#     n_jobs=-1,
#     verbose=1
# )

# grid_search.fit(X_train_scaled, y_train)
# best_rf = grid_search.best_estimator_
# print("Best parameters:", grid_search.best_params_)

# Train Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=2500,
    max_depth=None,
    min_samples_split=200,
    min_samples_leaf=100,
    max_features=0.4,
    bootstrap=True,
    max_samples=0.9,
    oob_score=True,
    criterion='squared_error',
    random_state=42
)

rf.fit(X_train_processed, y_train)

# Evaluate on validation set
y_pred = rf.predict(X_val_processed)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Train MSE vs formula: {mse:.4f}")
print(f"Train R² vs formula: {r2:.4f}")

# Evaluate on test set
df_test = filtering("./src/claims_test.csv")

X_test = df_test.drop(columns=["ClaimNb", "Exposure", "Risk"])
y_test = df_test["Risk"]

X_test_processed = preprocessor.transform(X_test)

y_pred_test = rf.predict(X_test_processed)
                                                                     
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Test MSE vs formula: {mse_test:.4f}")
print(f"Test R² vs formula: {r2_test:.4f}")