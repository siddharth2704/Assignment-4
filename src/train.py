import pandas as pd
import joblib
import argparse
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path

def main(params):
    df = pd.read_csv("data/data.csv")
    X = df[["feature"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["split"]["test_size"], random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ensure folders exist
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # Save model and test data
    joblib.dump(model, "models/model.pkl")
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    # Save predictions for evaluation
    preds = model.predict(X_test)
    pd.DataFrame({"pred": preds}).to_csv("outputs/y_pred.csv", index=False)

    print("Training complete. Model saved at models/model.pkl")

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    main(params)
