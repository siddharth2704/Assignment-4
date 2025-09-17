import pandas as pd
import json
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def main():
    y_true = pd.read_csv("data/processed/y_test.csv")["target"].to_numpy()
    y_pred = pd.read_csv("outputs/y_pred.csv")["pred"].to_numpy()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {"rmse": rmse, "mse": mse, "r2": r2}

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    main()
