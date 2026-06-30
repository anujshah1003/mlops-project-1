import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2
)

# Enable MLflow tracking
mlflow.start_run()

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Log parameters & metrics
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", acc)

# Log model
mlflow.sklearn.log_model(model, "model")

mlflow.end_run()

print(f"Accuracy: {acc}")