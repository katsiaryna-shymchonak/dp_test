import numpy as np
import pandas as pd
import pytest
from src.models import (
    generate_portfolio_forecasts,
    holt_winters_forecast,
    seasonal_naive_forecast,
    tsb_forecast,
    wma_forecast,
)


@pytest.fixture
def synthetic_time_series():
    """Synthetic 36-month time series with trend and seasonality."""
    np.random.seed(42)
    time = np.arange(36)
    seasonal = 50 * np.sin(2 * np.pi * time / 12)
    trend = 10 * time
    noise = np.random.normal(0, 5, 36)
    series = pd.Series(np.maximum(10, 200 + trend + seasonal + noise))
    return series


def test_baselines_forecast_length_and_non_negativity(synthetic_time_series):
    """Checks horizon dimensions and non-negativity for WMA and Seasonal Naive."""
    horizon = 3
    snaive_res = seasonal_naive_forecast(synthetic_time_series, horizon=horizon)
    wma_res = wma_forecast(synthetic_time_series, horizon=horizon)

    assert len(snaive_res) == horizon, "Invalid horizon for Seasonal Naive"
    assert len(wma_res) == horizon, "Invalid horizon for WMA"
    assert (snaive_res >= 0).all(), "Negative values in Seasonal Naive"
    assert (wma_res >= 0).all(), "Negative values in WMA"


def test_tsb_forecast_intermittent_data():
    """Checks TSB model stability on intermittent series with zeros."""
    intermittent_series = pd.Series([0, 0, 50, 0, 0, 0, 40, 0, 0, 0, 0, 30])
    horizon = 3

    tsb_res = tsb_forecast(intermittent_series, horizon=horizon)

    assert len(tsb_res) == horizon, "TSB forecast length does not match horizon"
    assert (tsb_res >= 0).all(), "TSB produced negative values"
    assert len(np.unique(tsb_res)) == 1, "TSB for intermittent demand should produce constant step across horizon"


def test_holt_winters_smooth_demand(synthetic_time_series):
    """Checks Holt-Winters forecast generation for smooth demand."""
    horizon = 3
    hw_res = holt_winters_forecast(synthetic_time_series, horizon=horizon)

    assert len(hw_res) == horizon
    assert not np.isnan(hw_res).any(), "Holt-Winters returned NaN"
    assert (hw_res >= 0).all(), "Holt-Winters returned negative values"


def test_generate_portfolio_forecasts_ensemble_blending():
    """Mathematical test for exact 50/50 ensemble blending (HW/TSB + SNaive)."""
    # Create 36 months of identical data for predictability
    dates = pd.date_range("2021-01-01", periods=36, freq="MS").strftime("%Y-%m").tolist()
    df = pd.DataFrame({
        "sku": ["SKU_TEST"] * 36,
        "location": ["MSK"] * 36,
        "period": dates,
        "qty": [100.0] * 36
    })

    sku_seg = pd.DataFrame({"sku": ["SKU_TEST"], "demand_type": ["Smooth"]})

    forecasts = generate_portfolio_forecasts(df, sku_seg, horizon=3)

    assert "forecast_ensemble" in forecasts.columns, "Column forecast_ensemble is missing"

    # Verify math: ensemble = 0.5 * primary + 0.5 * snaive
    row = forecasts.iloc[0]
    expected_ensemble = round(0.5 * row["forecast_primary"] + 0.5 * row["baseline_snaive"], 2)

    assert row["forecast_ensemble"] == pytest.approx(expected_ensemble, abs=0.01), \
        "Ensemble calculation does not match 50/50 formula"


def test_generate_portfolio_forecasts_schema_and_routing():
    """Checks output schema and routing logic of portfolio router."""
    dates = pd.date_range("2021-01-01", periods=36, freq="MS").strftime("%Y-%m").tolist()
    df = pd.DataFrame({
        "sku": ["SKU_SMOOTH"] * 36 + ["SKU_LUMPY"] * 36,
        "location": ["MSK"] * 72,
        "period": dates + dates,
        "qty": [100.0] * 36 + [0.0, 50.0, 0.0, 0.0] * 9
    })

    sku_seg = pd.DataFrame({
        "sku": ["SKU_SMOOTH", "SKU_LUMPY"],
        "demand_type": ["Smooth", "Lumpy"]
    })

    res = generate_portfolio_forecasts(df, sku_seg, horizon=3)

    assert len(res) == 6, "Expected 6 rows (2 SKUs * 3 horizon steps)"
    assert set(res["model_primary"].unique()) == {"Holt-Winters", "TSB"}, "Incorrect model routing"