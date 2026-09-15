from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.pipeline import run_forecasting_pipeline


@pytest.fixture
def mock_project_dir(tmp_path):
    """Creates a temporary project directory with dummy input CSV files (36 periods, 2 SKUs)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    periods = pd.date_range("2023-01-01", periods=36, freq="MS").strftime("%Y-%m").tolist()

    # Sales history (SKU_SMOOTH: regular demand, SKU_LUMPY: intermittent demand)
    sales_data = []
    for p in periods:
        sales_data.append({"sku": "SKU_SMOOTH", "location": "MSK", "period": p, "qty": 100.0})
        qty_lumpy = 50.0 if int(p[-2:]) % 3 == 0 else 0.0
        sales_data.append({"sku": "SKU_LUMPY", "location": "MSK", "period": p, "qty": qty_lumpy})

    sales_df = pd.DataFrame(sales_data)
    sales_df.to_csv(data_dir / "sales_history.csv", index=False)

    # Stock history
    stock_data = []
    for p in periods:
        stock_data.append({"sku": "SKU_SMOOTH", "location": "MSK", "period": p, "stock_end_qty": 200, "days_out_of_stock": 0})
        stock_data.append({"sku": "SKU_LUMPY", "location": "MSK", "period": p, "stock_end_qty": 100, "days_out_of_stock": 0})

    stock_df = pd.DataFrame(stock_data)
    stock_df.to_csv(data_dir / "stock_history.csv", index=False)

    # Item master
    items_df = pd.DataFrame({
        "sku": ["SKU_SMOOTH", "SKU_LUMPY"],
        "category": ["CatA", "CatB"],
        "abc_class": ["A", "C"],
        "launch_period": ["2023-01", "2023-01"],
        "predecessor_sku": [np.nan, np.nan],
        "uom": ["pcs", "pcs"]
    })
    items_df.to_csv(data_dir / "item_master.csv", index=False)

    # Promo calendar
    promos_df = pd.DataFrame({
        "sku": ["SKU_SMOOTH"],
        "period": ["2023-05"],
        "promo_type": ["Discount"],
        "discount_pct": [15.0]
    })
    promos_df.to_csv(data_dir / "promo_calendar.csv", index=False)

    return tmp_path


def test_pipeline_end_to_end_execution(mock_project_dir):
    """Checks full execution of run_forecasting_pipeline on temporary data."""
    forecast_df = run_forecasting_pipeline(mock_project_dir, horizon=3)

    assert isinstance(forecast_df, pd.DataFrame), "Pipeline output is not a DataFrame"
    assert not forecast_df.empty, "Pipeline returned empty DataFrame"
    assert len(forecast_df) == 6, "Expected 6 rows (2 SKUs * 3 horizon steps)"


def test_pipeline_output_files_generated(mock_project_dir):
    """Checks that all required output CSV files and folders are created."""
    run_forecasting_pipeline(mock_project_dir, horizon=3)

    outputs_dir = mock_project_dir / "outputs"
    processed_dir = mock_project_dir / "data" / "processed"

    assert outputs_dir.exists(), "outputs/ directory was not created"
    assert (outputs_dir / "forecast_results.csv").exists(), "forecast_results.csv is missing"
    assert (outputs_dir / "model_evaluation_metrics.csv").exists(), "model_evaluation_metrics.csv is missing"
    assert (outputs_dir / "final_submission.csv").exists(), "final_submission.csv is missing"
    assert (processed_dir / "cleaned_demand_features.csv").exists(), "cleaned_demand_features.csv is missing"


def test_pipeline_submission_format_and_dates(mock_project_dir):
    """Checks target schema, dates (2026-01..2026-03), and non-negativity of final submission."""
    run_forecasting_pipeline(mock_project_dir, horizon=3)

    submission_path = mock_project_dir / "outputs" / "final_submission.csv"
    submission_df = pd.read_csv(submission_path)

    expected_cols = ["sku", "location", "period", "forecast_qty"]
    assert list(submission_df.columns) == expected_cols, "Submission columns do not match requirements"

    expected_periods = {"2026-01", "2026-02", "2026-03"}
    assert set(submission_df["period"].unique()) == expected_periods, "Target forecast periods do not match horizon"

    assert submission_df["forecast_qty"].isna().sum() == 0, "Submission contains NaN values"
    assert (submission_df["forecast_qty"] >= 0).all(), "Submission contains negative forecast values"