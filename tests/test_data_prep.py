import numpy as np
import pandas as pd
import pytest
from src.data_prep import (
    clean_data,
    clean_sales_records,
    fix_ghost_stock,
    map_predecessors,
    restore_demand,
)


@pytest.fixture
def sample_sales_df():
    """Fixture for raw sales data with defects (duplicates and negative returns)."""
    return pd.DataFrame(
        {
            "sku": ["SKU_1", "SKU_1", "SKU_1", "SKU_2"],
            "location": ["MSK", "MSK", "MSK", "MSK"],
            "period": ["2023-01", "2023-01", "2023-02", "2023-01"],
            "qty": [100.0, 100.0, -15.0, 50.0],
        }
    )


@pytest.fixture
def sample_stock_df():
    """Fixture for stock history with Ghost Stock defect."""
    return pd.DataFrame(
        {
            "sku": ["SKU_1", "SKU_1", "SKU_2"],
            "location": ["MSK", "MSK", "MSK"],
            "period": ["2023-01", "2023-02", "2023-01"],
            "stock_end_qty": [150, 0, 200],
            "days_out_of_stock": [0, 0, 0],  # Ghost stock on SKU_1 (2023-02)
        }
    )


def test_clean_sales_records_deduplication(sample_sales_df):
    """Checks deduplication of sales records by [sku, location, period] key."""
    cleaned = clean_sales_records(sample_sales_df)

    assert len(cleaned) == 3, "Duplicate records were not removed"
    assert (
        cleaned.duplicated(subset=["sku", "location", "period"]).sum() == 0
    ), "Duplicate keys remaining in dataset"


def test_clean_sales_records_negative_clipping(sample_sales_df):
    """Checks clipping of negative sales volumes (returns) to zero."""
    cleaned = clean_sales_records(sample_sales_df)
    negative_qty_count = (cleaned["qty"] < 0).sum()

    assert negative_qty_count == 0, "Negative sales were not clipped"
    assert (
        cleaned.loc[cleaned["period"] == "2023-02", "qty"].values[0] == 0.0
    ), "Return was not replaced with 0"


def test_fix_ghost_stock(sample_stock_df):
    """Checks detection of Ghost Stock (zero stock with 0 OOS days)."""
    fixed_stock = fix_ghost_stock(sample_stock_df)

    ghost_row = fixed_stock[
        (fixed_stock["sku"] == "SKU_1") & (fixed_stock["period"] == "2023-02")
    ]
    assert (
        ghost_row["days_out_of_stock"].values[0] == 30
    ), "Ghost Stock was not adjusted to 30 OOS days"


def test_map_predecessors():
    """Checks remapping of history from old SKU to new SKU (Cold Start)."""
    df = pd.DataFrame(
        {
            "sku": ["OLD_SKU_1", "REGULAR_SKU"],
            "location": ["MSK", "MSK"],
            "period": ["2023-01", "2023-01"],
            "qty": [100, 200],
        }
    )
    items = pd.DataFrame(
        {
            "sku": ["NEW_SKU_1", "REGULAR_SKU"],
            "predecessor_sku": ["OLD_SKU_1", np.nan],
        }
    )

    mapped = map_predecessors(df, items)

    assert "OLD_SKU_1" not in mapped["sku"].values, "Old SKU remained in sku column"
    assert "NEW_SKU_1" in mapped["sku"].values, "New SKU was not applied from mapping"


def test_restore_demand_partial_and_full_oos():
    """Checks proportional restoration logic and organic median imputation."""
    df = pd.DataFrame(
        {
            "sku": ["SKU_1", "SKU_1", "SKU_1"],
            "location": ["MSK", "MSK", "MSK"],
            "period": ["2023-01", "2023-02", "2023-03"],
            "qty": [100.0, 15.0, 0.0],
            "days_out_of_stock": [
                0,
                15,
                30,
            ],  # 0 days, 15 days (partial), 30 days (full)
            "discount_pct": [0.0, 0.0, 0.0],
        }
    )

    restored = restore_demand(df)
    qty_res = restored["qty_restored"].values

    # No OOS -> 100
    assert qty_res[0] == 100.0
    # Partial OOS (15 days): 15 / (30 - 15) * 30 = 30.0
    assert qty_res[1] == 30.0
    # Full OOS (30 days): clean organic median imputation (100.0)
    assert qty_res[2] == 100.0


def test_clean_data_end_to_end(sample_sales_df, sample_stock_df):
    """End-to-end integration test for clean_data pipeline."""
    items = pd.DataFrame({"sku": ["SKU_1", "SKU_2"]})
    promos = pd.DataFrame(
        {"sku": ["SKU_1"], "period": ["2023-01"], "discount_pct": [25.0]}
    )

    cleaned_df = clean_data(sample_sales_df, sample_stock_df, items, promos)

    required_cols = ["sku", "location", "period", "qty", "days_out_of_stock"]
    for col in required_cols:
        assert col in cleaned_df.columns, f"Column {col} is missing from feature matrix"

    assert cleaned_df["qty"].isna().sum() == 0, "Feature matrix contains NaN values in qty"
    assert (cleaned_df["qty"] >= 0).all(), "Negative sales present in feature matrix"