import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml

params = yaml.safe_load(open("params.yaml"))

df = pd.read_csv("data/processed.csv")

X = df.drop("target", axis=1)
y = df["target"]

model = RandomForestClassifier(n_estimators=params["train"]["n_estimators"])
model.fit(X, y)

joblib.dump(model, "models/model.pkl")