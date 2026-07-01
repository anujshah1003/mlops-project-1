import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import mlflow
# read saved run_id
with open("artifacts/run_id.txt") as f:
    run_id = f.read().strip()
    # resume the same run
mlflow.start_run(run_name=run_id)

df = pd.read_csv("data/processed.csv")

X = df.drop("target", axis=1)
y = df["target"]

model = joblib.load("models/model.pkl")

preds = model.predict(X)
acc = accuracy_score(y, preds)

with open("metrics.txt", "w") as f:
    f.write(f"accuracy: {acc}")

mlflow.log_metric("accuracy", acc)
mlflow.log_artifact("metrics.txt")
# mlflow.log_artifact("confusion_matrix.png")
# mlflow.log_artifact("classification_report.txt")
mlflow.end_run()