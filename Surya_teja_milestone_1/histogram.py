import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

cols = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI"]

plt.figure(figsize=(14,12))

for i, col in enumerate(cols, 1):
    plt.subplot(4,3,i)
    plt.hist(df[col])
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Freq")
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.show()
