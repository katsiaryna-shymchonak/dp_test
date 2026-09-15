import numpy as np
import pandas as pd
from src.models import generate_portfolio_forecasts, lightgbm_tweedie_forecast


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Weighted Absolute Percentage Error (WAPE)."""
    sum_true = np.sum(y_true)
    if sum_true == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / sum_true * 100)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 12,
) -> float:
    """Calculates Mean Absolute Scaled Error (MASE) against in-sample seasonal naive baseline."""
    mae_model = np.mean(np.abs(y_true - y_pred))

    if len(y_train) <= seasonality:
        mae_naive = np.mean(np.abs(np.diff(y_train)))
    else:
        mae_naive = np.mean(
            np.abs(y_train[seasonality:] - y_train[:-seasonality])
        )

    if mae_naive == 0 or np.isnan(mae_naive):
        return 1.0

    return float(mae_model / mae_naive)


def evaluate_holdout_performance(
    df: pd.DataFrame, sku_segmentation: pd.DataFrame, test_periods: int = 3
) -> pd.DataFrame:
    """Performs out-of-sample backtest on the last test_periods months to evaluate all models including Ensemble."""
    periods = sorted(df["period"].unique())
    train_periods = periods[:-test_periods]
    test_periods_list = periods[-test_periods:]

    train_df = df[df["period"].isin(train_periods)].copy()
    test_df = df[df["period"].isin(test_periods_list)].copy()

    fc_results = generate_portfolio_forecasts(
        df=train_df, sku_segmentation=sku_segmentation, horizon=test_periods
    )
    lgb_fc = lightgbm_tweedie_forecast(
        train_df=train_df,
        target_skus=train_df["sku"].unique(),
        horizon=test_periods,
    )

    preds_df = fc_results.merge(
        lgb_fc,
        left_on=["sku", "horizon_step"],
        right_on=["sku", "step"],
        how="left",
    ).drop(columns=["step"])

    period_map = {p: idx + 1 for idx, p in enumerate(test_periods_list)}
    test_df["horizon_step"] = test_df["period"].map(period_map)

    merged = test_df.merge(preds_df, on=["sku", "horizon_step"], how="inner")

    model_cols = {
        "forecast_ensemble": "Ensemble (HW/TSB + SNaive)",
        "forecast_primary": "Primary Champion",
        "forecast_sarima": "Auto-SARIMA",
        "baseline_wma": "Baseline WMA",
        "baseline_snaive": "Seasonal Naive",
        "forecast_lgb": "LightGBM Tweedie",
    }

    metrics_list = []
    for sku, group in merged.groupby("sku"):
        train_series = train_df[train_df["sku"] == sku]["qty"].values
        actuals = group["qty"].values

        for col, model_name in model_cols.items():
            if col in group.columns:
                preds = group[col].values
                metrics_list.append(
                    {
                        "sku": sku,
                        "model": model_name,
                        "wape": calculate_wape(actuals, preds),
                        "rmse": calculate_rmse(actuals, preds),
                        "mase": calculate_mase(
                            actuals, preds, train_series, seasonality=12
                        ),
                    }
                )

    results_df = pd.DataFrame(metrics_list)
    summary = (
        results_df.groupby("model")
        .agg({"wape": "mean", "rmse": "mean", "mase": "mean"})
        .reset_index()
    )
    return summary.sort_values(by="wape").round(2)