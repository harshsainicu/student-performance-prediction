from pathlib import Path
import pandas as pd

def test_dataset_has_expected_columns():
    path = Path(__file__).resolve().parents[1] / "data" / "student_scores.csv"
    data = pd.read_csv(path)
    expected = {"study_hours", "attendance", "previous_score", "final_score"}
    assert expected.issubset(data.columns)
    assert len(data) >= 10
    assert data[list(expected)].notna().all().all()
