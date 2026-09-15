import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure root directory is accessible in pytest paths
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import run_forecasting_pipeline


def test_pipeline_end_to_end_execution(tmp_path):
    """Verifies that the end-to-end pipeline executes without throwing exceptions."""
    # Run forecasting pipeline using current workspace data
    forecasts = run_forecasting_pipeline(project_root, horizon=3)
    assert not forecasts.empty
    assert "forecast_ensemble" in forecasts.columns


def test_pipeline_output_files_generated(tmp_path):
    """Verifies that all required artifact CSV files are created in outputs/ and data/processed/."""
    run_forecasting_pipeline(project_root, horizon=3)

    assert (project_root / "data" / "processed" / "cleaned_demand_features.csv").exists()
    assert (project_root / "outputs" / "forecast_results.csv").exists()
    assert (project_root / "outputs" / "model_evaluation_metrics.csv").exists()
    assert (project_root / "outputs" / "final_submission.csv").exists()


def test_pipeline_submission_format_and_dates(tmp_path):
    """Verifies final submission schema, non-null predictions, and target period formatting."""
    run_forecasting_pipeline(project_root, horizon=3)

    submission_df = pd.read_csv(project_root / "outputs" / "final_submission.csv")

    # Strict check for mandatory 3-column business schema (without location)
    expected_columns = ["sku", "period", "forecast_qty"]
    assert list(submission_df.columns) == expected_columns, "Submission columns do not match requirements"

    # Verify forecast values and dates
    assert not submission_df["forecast_qty"].isnull().any()
    assert (submission_df["forecast_qty"] >= 0).all()
    assert set(submission_df["period"].unique()) == {"2026-01", "2026-02", "2026-03"}
