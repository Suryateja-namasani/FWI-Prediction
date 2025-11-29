import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("forestfires_final_clean.csv")

X = df[['month','day','ffmc','dmc','dc','isi','temp','wind','bui']]
y = df['fwi']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

alphas = [0.0001,0.001,0.01,0.1,1,10,100]

train_mse_list = []
test_mse_list = []
train_rmse_list = []
test_rmse_list = []
train_mae_list = []
test_mae_list = []

best_alpha = None
best_test_mse = float("inf")

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    train_rmse = math.sqrt(train_mse)
    test_rmse = math.sqrt(test_mse)

    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)
    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)
    train_mae_list.append(train_mae)
    test_mae_list.append(test_mae)

    if test_mse < best_test_mse:
        best_test_mse = test_mse
        best_alpha = alpha

print("Best alpha:", best_alpha)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(alphas, train_mse_list, marker='o', label='Train')
plt.plot(alphas, test_mse_list, marker='o', label='Test')
plt.xscale('log')
plt.xlabel("alpha")
plt.ylabel("MSE")
plt.title("MSE vs Alpha")
plt.legend()
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(alphas, train_rmse_list, marker='o', label='Train')
plt.plot(alphas, test_rmse_list, marker='o', label='Test')
plt.xscale('log')
plt.xlabel("alpha")
plt.ylabel("RMSE")
plt.title("RMSE vs Alpha")
plt.legend()
plt.grid(True)

plt.subplot(1,3,3)
plt.plot(alphas, train_mae_list, marker='o', label='Train')
plt.plot(alphas, test_mae_list, marker='o', label='Test')
plt.xscale('log')
plt.xlabel("alpha")
plt.ylabel("MAE")
plt.title("MAE vs Alpha")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
