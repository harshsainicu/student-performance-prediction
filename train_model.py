"""Train and compare two beginner regression models."""

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_scores.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FEATURES = ["study_hours", "attendance", "previous_score"]
TARGET = "final_score"


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
            "r2": round(float(r2_score(y_test, predictions)), 3),
        }
        trained_models[name] = model
        print(f"{name}: MAE={results[name]['mae']}, R²={results[name]['r2']}")

    # The app uses Linear Regression because it is easy to explain.
    joblib.dump(trained_models["Linear Regression"], MODEL_DIR / "linear_regression.pkl")
    joblib.dump(trained_models["Decision Tree"], MODEL_DIR / "decision_tree.pkl")
    (REPORT_DIR / "model_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Save a simple chart comparing the two models.
    names = list(results.keys())
    mae_values = [results[name]["mae"] for name in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, mae_values)
    ax.set_title("Model Comparison: Mean Absolute Error")
    ax.set_ylabel("MAE (lower is better)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    print("\nTraining complete.")
    print(f"Models saved in: {MODEL_DIR}")
    print(f"Results saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
