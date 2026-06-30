import pandas as pd

df = pd.read_csv("data/raw.csv")

# Simple preprocessing
df = df.dropna()

df.to_csv("data/processed.csv", index=False)