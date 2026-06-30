import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/processed.csv")

X = df.drop("target", axis=1)
y = df["target"]

model = joblib.load("models/model.pkl")

preds = model.predict(X)
acc = accuracy_score(y, preds)

with open("metrics.txt", "w") as f:
    f.write(f"accuracy: {acc}")