import pandas as pd
import mlflow
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib

df = pd.read_csv("data/training_data.csv")

X = df.drop("build_time_min", axis=1)
y = df["build_time_min"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("pipelineiq-build-time-min")

results = []

models = {
    "SVR": SVR(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("domain", "ci_cd")

        results.append({"name": name, "mae": mae, "rmse": rmse})

        joblib.dump(model, f"models/{name}.pkl")

best = min(results, key=lambda x: x["mae"])

output = {
    "experiment_name": "pipelineiq-build-time-min",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)
