'''Feed-Forward Neural Network regressor - reference implementation'''

# ...

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

df_original=pd.read_csv('src\claims_train.csv')
#Since the density is the continiuous variable, we can drop the Area column
df_original.drop('Area', axis=1, inplace=True)
#Rescale BonusMalus, since in the literature it is said to be between 0.5 and 3.5
df_original['BonusMalus']=df_original['BonusMalus']/100
#Filtering the Exposure since it goes out of bounds (0-1)
df = df_original[df_original['Exposure']<=1]
print(df.shape)
#Filterig VehAge to be less than 25, since on the roads these are the most common vehicles
df = df[(df['VehAge']<=25)]
print(df.shape)

#After inspection of the Density by Region boxplots, we decided to remove outliers only for specific regions 
#(the main reasons being that some regions had extreme outliers that cannot be explained by real world data, while others were relatively clean.)
regions_remove_outliers = ['R25','R82','R54','R94','R93','R91','R52',
                            'R72','R31','R73','R23','R22','R41','R42',
                            'R83','R21','R74','R43', 'R11']

def remove_outliers_selective(group):
    # Only remove outliers for regions in the list
    if group.name in regions_remove_outliers:
        q1, q3 = group['Density'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return group[(group['Density'] >= lower) & (group['Density'] <= upper)]
    else:
        return group

df = df.groupby('Region', group_keys=False).apply(remove_outliers_selective)
print(df.shape)
#Let's fix the BonusMalus, since only those could have 0.5 who had 13 years of accident free driving. that means until the age of 31
#nobody can have malus 0.5 however there is no limit for the top value (The overall top limit is 3.50, bottom limit 0.5)
#BonusMalus= PreviousMalus * 0.95 if no accident else 1.25
#We replaced the incorrect values with the average value of the age
min_malus = 0.95 ** (df['DrivAge']-18)
bad_mal_mask = df['BonusMalus'] >= min_malus
age_avg = df[bad_mal_mask].groupby('DrivAge')['BonusMalus'].mean()
impossible_malus_mask = df['BonusMalus']<0.95**(df['DrivAge']-18)
df.loc[impossible_malus_mask, 'BonusMalus'] = df.loc[impossible_malus_mask, 'DrivAge'].map(age_avg)

alpha=2 # If we want to penaltize the claimnb more
beta=0 # If we want to finetune the exposure part 
gamma=0.1 # To avoid log(0)
df['Risk'] = (np.log(1+(gamma+df['ClaimNb']**alpha)/(df['Exposure']+beta))/(1+(np.log(1+(gamma+df['ClaimNb']**alpha)/(df['Exposure']+beta)))))

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