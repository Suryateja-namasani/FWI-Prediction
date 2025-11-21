import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset.csv")

df = df.fillna(0)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype("category").cat.codes

cols = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]

corr = df[cols].corr()
print(corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap with Values")
plt.tight_layout()
plt.show()
