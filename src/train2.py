import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml
import mlflow

mlflow.start_run()
run_id = mlflow.active_run().info.run_id
print("RUN_ID:", run_id)
with open("artifacts/run_id.txt", "w") as f:
    f.write(f"run_id: {run_id}")

params = yaml.safe_load(open("params.yaml"))

df = pd.read_csv("data/processed.csv")

X = df.drop("target", axis=1)
y = df["target"]

model = RandomForestClassifier(n_estimators=params["train"]["n_estimators"])
model.fit(X, y)

joblib.dump(model, "models/model.pkl")

mlflow.log_param("n_estimators", params["train"]["n_estimators"])

mlflow.sklearn.log_model(model, "model")

mlflow.end_run()
