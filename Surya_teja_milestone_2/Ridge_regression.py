import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv(r"D:\wfi\FWI-Prediction\FWI-Prediction\dataset.csv")

# Strip column names and remove spaces
df.columns = df.columns.str.strip()

# Convert numeric columns to float
numeric_cols = ['FFMC','DMC','DC','ISI','Temperature','Ws','BUI','FWI']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')

# Drop NaN rows
df = df.dropna()

# Features and target
X = df[['FFMC','DMC','DC','ISI','Temperature','Ws','BUI']].values
y = df['FWI'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
with open("scaler.pkl","wb") as f:
    pickle.dump(scaler, f)

# Ridge regression alpha tuning
alpha_values = [0.01, 0.1, 1, 10, 100]

mse_train_list, mse_test_list = [], []
mae_list, rmse_list = [], []

best_r2 = -np.inf
best_alpha = None

for alpha in alpha_values:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mse_test)
    r2 = r2_score(y_test, y_test_pred)

    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)
    mae_list.append(mae)
    rmse_list.append(rmse)

    print(f"\nAlpha: {alpha}")
    print("Train MSE:", mse_train)
    print("Test MSE:", mse_test)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    if r2 > best_r2:
        best_r2 = r2
        best_alpha = alpha
        best_model = model

# Save best model
with open("ridge.pkl","wb") as f:
    pickle.dump(best_model, f)

print("\nBest alpha:", best_alpha)

 

# Plots
plt.figure()
plt.plot(alpha_values, mse_train_list, marker='o', label="Train MSE")
plt.plot(alpha_values, mse_test_list, marker='o', label="Test MSE")
plt.xscale('log')
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("MSE vs Alpha")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(alpha_values, mae_list, marker='o')
plt.xscale('log')
plt.xlabel("Alpha")
plt.ylabel("MAE")
plt.title("MAE vs Alpha")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(alpha_values, rmse_list, marker='o')
plt.xscale('log')
plt.xlabel("Alpha")
plt.ylabel("RMSE")
plt.title("RMSE vs Alpha")
plt.grid(True)
plt.show()

# Predicted vs Actual
y_pred_best = best_model.predict(X_test)
plt.figure()
plt.scatter(y_test, y_pred_best)
plt.xlabel("Actual FWI")
plt.ylabel("Predicted FWI")
plt.title("Predicted vs Actual")
plt.grid(True)
plt.show()
