'''Decision Tree regressor with at least one categorical variable - reference implementation'''

import numpy as np
import pandas as pd
from filter_data import filtering
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

data = pd.read_csv('/Users/swenpai/machine-n-learnings-inc/src/claims_train.csv')
df = filtering(data)

features = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
target = "risk"

X = df[features]
y = df[target]

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train decision tree
tree = DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=40,
    min_samples_leaf=10,
    random_state=42
)

tree.fit(X_train, y_train)

# Prediction
y_pred = tree.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
print("Test MSE:", mse)

scores = cross_val_score(
    tree,
    X,
    y,
    scoring="neg_mean_squared_error",
    cv=5
)

print("Average CV MSE:", -scores.mean())

path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

best_mse = float("inf")
best_alpha = 0

for alpha in ccp_alphas:
    pruned_tree = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        ccp_alpha=alpha
    )
    pruned_tree.fit(X_train, y_train)
    preds = pruned_tree.predict(X_val)
    mse = mean_squared_error(y_val, preds)

    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha

print("Best alpha:", best_alpha)

# Train final pruned model
final_tree = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    ccp_alpha=best_alpha
)
final_tree.fit(X_train, y_train)