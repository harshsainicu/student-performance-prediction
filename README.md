# Student Performance Prediction (Beginner ML Project)

A simple machine-learning assignment that predicts a student's final score from a few study-related inputs. It compares a Linear Regression model with a Decision Tree Regressor and includes a small Streamlit app.

**Project level:** Beginner  
**Language:** Python  
**Dataset:** A small sample CSV included in `data/student_scores.csv` so the project can run immediately.

> This is a learning demo using a tiny illustrative dataset, not a reliable tool for judging real students.

## Features
- Read a CSV file with pandas
- Explore study hours, attendance, and previous score
- Train two beginner-friendly regression models
- Compare models with MAE and R²
- Predict a score from inputs in a simple web app
- Save a comparison chart and model files

## Setup (Windows)
Open PowerShell in this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py` does not work, try `python` instead.

## Run training
```powershell
python train_model.py
```

This creates files in `models/` and `reports/`.

## Run the app
```powershell
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.

## Project files
```text
student-performance-prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── student_scores.csv
├── models/                 # created after training
├── reports/                # created after training
└── tests/
    └── test_features.py
```

## How it works
1. The dataset contains study hours, attendance percentage, previous score, and final score.
2. The program separates inputs (`study_hours`, `attendance`, `previous_score`) from the target (`final_score`).
3. It splits the rows into training and test sets.
4. It trains Linear Regression and Decision Tree Regression models.
5. It reports Mean Absolute Error (MAE) and R² on the test split.
6. The Streamlit app uses the saved Linear Regression model for a sample prediction.

## Important note
The included dataset is intentionally small and illustrative for a beginner assignment. Metrics can change with the split and should not be interpreted as evidence of real-world accuracy.
