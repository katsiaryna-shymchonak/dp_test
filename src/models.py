import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb

warnings.filterwarnings("ignore")


# Baseline models


def seasonal_naive_forecast(
    series: pd.Series, horizon: int = 3, season_length: int = 12
) -> np.ndarray:
    """Repeats the last observed seasonal cycles for the forecast horizon."""
    values = series.values
    if len(values) < season_length:
        return np.full(horizon, values[-1] if len(values) > 0 else 0.0)

    forecast = []
    for h in range(horizon):
        lag_idx = -season_length + (h % season_length)
        forecast.append(values[lag_idx])
    return np.array(forecast)


def wma_forecast(
    series: pd.Series, horizon: int = 3, weights: list = [0.5, 0.3, 0.2]
) -> np.ndarray:
    """Calculates 3-month Weighted Moving Average as a static step-out forecast."""
    values = series.values
    window = len(weights)
    if len(values) < window:
        return np.full(horizon, values.mean() if len(values) > 0 else 0.0)

    weights_arr = np.array(weights) / sum(weights)
    recent_values = values[-window:]
    wma_value = np.sum(recent_values * weights_arr[::-1])
    return np.full(horizon, max(0.0, wma_value))


# Statistical model engines


def holt_winters_forecast(
    series: pd.Series, horizon: int = 3, seasonal_periods: int = 12
) -> np.ndarray:
    """Triple Exponential Smoothing for smooth demand with trend and annual seasonality."""
    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit()
        forecast = model.forecast(horizon).values
        return np.maximum(0.0, forecast)
    except Exception:
        return wma_forecast(series, horizon=horizon)


def auto_sarima_forecast(
    series: pd.Series, horizon: int = 3, seasonal_periods: int = 12
) -> np.ndarray:
    """SARIMAX engine serving as a statistical benchmark for complex autocorrelation."""
    try:
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, seasonal_periods),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        forecast = model.forecast(steps=horizon).values
        return np.maximum(0.0, forecast)
    except Exception:
        return holt_winters_forecast(
            series, horizon=horizon, seasonal_periods=seasonal_periods
        )


def tsb_forecast(
    series: pd.Series,
    horizon: int = 3,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> np.ndarray:
    """Teunter-Syntetos-Babai (TSB) method for intermittent and erratic SKUs."""
    y = series.values
    n = len(y)

    non_zero = y[y > 0]
    if len(non_zero) == 0:
        return np.zeros(horizon)

    p = np.zeros(n)
    z = np.zeros(n)

    p[0] = len(non_zero) / n
    z[0] = np.mean(non_zero)

    for t in range(1, n):
        if y[t] > 0:
            p[t] = p[t - 1] + alpha * (1.0 - p[t - 1])
            z[t] = z[t - 1] + beta * (y[t] - z[t - 1])
        else:
            p[t] = p[t - 1] + alpha * (0.0 - p[t - 1])
            z[t] = z[t - 1]

    final_forecast = p[-1] * z[-1]
    return np.full(horizon, max(0.0, final_forecast))


# Optional ML fallback


def prepare_lgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates lag and rolling aggregation features for temporal ML modeling."""
    df_feat = df.sort_values(["sku", "period"]).copy()

    for lag in [1, 2, 3, 12]:
        df_feat[f"lag_{lag}"] = df_feat.groupby("sku")["qty"].shift(lag)

    df_feat["rolling_mean_3"] = df_feat.groupby("sku")["qty"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    df_feat["rolling_std_3"] = df_feat.groupby("sku")["qty"].transform(
        lambda x: x.shift(1).rolling(3).std()
    )

    return df_feat.dropna()


def lightgbm_tweedie_forecast(
    train_df: pd.DataFrame, target_skus: list, horizon: int = 3
) -> pd.DataFrame:
    """Global LightGBM model trained with Tweedie loss to natively handle zero-inflation."""
    featured_df = prepare_lgb_features(train_df)

    features = [
        c for c in featured_df.columns if c.startswith(("lag_", "rolling_"))
    ]
    X = featured_df[features]
    y = featured_df["qty"]

    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=100)

    predictions = []
    for sku in target_skus:
        sku_data = featured_df[featured_df["sku"] == sku].tail(1)
        if not sku_data.empty:
            X_test = sku_data[features]
            pred = model.predict(X_test)[0]
            for h in range(1, horizon + 1):
                predictions.append(
                    {"sku": sku, "step": h, "forecast_lgb": round(max(0.0, pred), 2)}
                )

    return pd.DataFrame(predictions)


# Portfolio router


def generate_portfolio_forecasts(
    df: pd.DataFrame, sku_segmentation: pd.DataFrame, horizon: int = 3
) -> pd.DataFrame:
    """Main orchestration router for statistical models, ensemble, and baselines."""
    results = []

    for sku, group in df.groupby("sku"):
        group = group.sort_values("period")
        series = group["qty"]

        demand_type_match = sku_segmentation.loc[
            sku_segmentation["sku"] == sku, "demand_type"
        ].values
        demand_type = (
            demand_type_match[0] if len(demand_type_match) > 0 else "Smooth"
        )

        s_naive = seasonal_naive_forecast(series, horizon=horizon)
        wma = wma_forecast(series, horizon=horizon)

        if demand_type == "Smooth":
            primary_fc = holt_winters_forecast(series, horizon=horizon)
            benchmark_fc = auto_sarima_forecast(series, horizon=horizon)
            model_used = "Holt-Winters"
        else:
            primary_fc = tsb_forecast(series, horizon=horizon)
            benchmark_fc = primary_fc
            model_used = "TSB"

        # Ensemble: 50% primary champion + 50% seasonal naive
        ensemble_fc = np.maximum(0.0, 0.5 * primary_fc + 0.5 * s_naive)

        for h in range(horizon):
            results.append(
                {
                    "sku": sku,
                    "demand_type": demand_type,
                    "horizon_step": h + 1,
                    "model_primary": model_used,
                    "forecast_primary": round(primary_fc[h], 2),
                    "forecast_ensemble": round(ensemble_fc[h], 2),
                    "forecast_sarima": round(benchmark_fc[h], 2),
                    "baseline_wma": round(wma[h], 2),
                    "baseline_snaive": round(s_naive[h], 2),
                }
            )

    return pd.DataFrame(results)