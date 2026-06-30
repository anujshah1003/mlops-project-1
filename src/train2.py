import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/processed.csv")

X = df.drop("target", axis=1)
y = df["target"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "models/model.pkl")