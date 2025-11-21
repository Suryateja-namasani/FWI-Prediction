import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")
print(df)

# check for nan values
print(df.isna().sum())

# replace nan with 0
df = df.fillna(0)

# convert categorical columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

print(df.head())
