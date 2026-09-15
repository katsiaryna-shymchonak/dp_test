from pathlib import Path
import pandas as pd
from src.data_prep import clean_data
from src.evaluation import evaluate_holdout_performance
from src.models import generate_portfolio_forecasts, lightgbm_tweedie_forecast


def run_forecasting_pipeline(
    project_root: Path, horizon: int = 3
) -> pd.DataFrame:
    """End-to-end execution pipeline connecting data prep, evaluation, and forecasting."""
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"

    # Load raw datasets from data/
    print("Loading raw datasets from data/...")
    sales = pd.read_csv(data_dir / "sales_history.csv")
    stock = pd.read_csv(data_dir / "stock_history.csv")
    items = pd.read_csv(data_dir / "item_master.csv")
    promos = pd.read_csv(data_dir / "promo_calendar.csv")

    # Run data preparation pipeline
    print("Running data preparation pipeline...")
    df_cleaned = clean_data(sales, stock, items, promos)

    # Compute ADI & CV^2 SKU demand segmentation dynamically
    print("Performing ADI & CV^2 SKU demand segmentation...")
    sku_stats = (
        df_cleaned.groupby("sku")["qty"]
        .agg(
            total_periods="count",
            active_periods=lambda x: (x > 0).sum(),
            mean_qty="mean",
            std_qty="std",
        )
        .reset_index()
    )

    sku_stats["ADI"] = sku_stats["total_periods"] / sku_stats["active_periods"]
    sku_stats["CV2"] = (sku_stats["std_qty"] / sku_stats["mean_qty"]) ** 2

    def categorize_demand(row):
        if row["ADI"] < 1.32 and row["CV2"] < 0.49:
            return "Smooth"
        elif row["ADI"] >= 1.32 and row["CV2"] < 0.49:
            return "Intermittent"
        elif row["ADI"] < 1.32 and row["CV2"] >= 0.49:
            return "Erratic"
        else:
            return "Lumpy"

    sku_stats["demand_type"] = sku_stats.apply(categorize_demand, axis=1)

    # Perform backtest evaluation on holdout set (last test_periods)
    print("Running holdout backtest evaluation...")
    metrics_summary = evaluate_holdout_performance(
        df=df_cleaned, sku_segmentation=sku_stats, test_periods=horizon
    )

    # Route SKUs to modeling engines for full horizon out-of-sample forecast
    print("Generating portfolio forecasts...")
    forecast_results = generate_portfolio_forecasts(
        df=df_cleaned, sku_segmentation=sku_stats, horizon=horizon
    )

    # Log model routing selection
    print("Model routing selection applied:")
    routing_summary = (
        forecast_results[forecast_results["horizon_step"] == 1]["model_primary"]
        .value_counts()
        .to_dict()
    )
    for model_name, count in routing_summary.items():
        print(f"  - {model_name}: {count} SKUs")
    print("  - Final Champion Ensemble (50% Primary + 50% S.Naive) applied to all SKUs.")

    # Secondary experiment: Global LightGBM with Tweedie loss
    print("Running secondary experiment (LightGBM Tweedie)...")
    lgb_forecasts = lightgbm_tweedie_forecast(
        train_df=df_cleaned,
        target_skus=df_cleaned["sku"].unique(),
        horizon=horizon,
    )

    # Merge LightGBM predictions as an auxiliary benchmark column
    final_forecasts = forecast_results.merge(
        lgb_forecasts,
        left_on=["sku", "horizon_step"],
        right_on=["sku", "step"],
        how="left",
    ).drop(columns=["step"])

    # Map horizon steps to target calendar periods (2026-01 ... 2026-03)
    horizon_date_map = {1: "2026-01", 2: "2026-02", 3: "2026-03"}
    final_forecasts["period"] = final_forecasts["horizon_step"].map(
        horizon_date_map
    )

    # Save outputs: internal features stay in data/processed, predictions go to outputs/
    print("Saving pipeline outputs...")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df_cleaned.to_csv(
        processed_dir / "cleaned_demand_features.csv", index=False
    )
    final_forecasts.to_csv(outputs_dir / "forecast_results.csv", index=False)
    metrics_summary.to_csv(
        outputs_dir / "model_evaluation_metrics.csv", index=False
    )

    # Save business final submission CSV (sku, location, period, forecast_qty)
    submission_df = final_forecasts[
        ["sku", "period", "forecast_ensemble"]
    ].rename(columns={"forecast_ensemble": "forecast_qty"})
    submission_df["location"] = "MSK"
    submission_df = submission_df[
        ["sku", "location", "period", "forecast_qty"]
    ]
    submission_df.to_csv(outputs_dir / "final_submission.csv", index=False)

    print("\n--- Model Evaluation Metrics (Holdout Backtest) ---")
    print(metrics_summary.to_string(index=False))

    return final_forecasts


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    forecasts = run_forecasting_pipeline(project_root, horizon=3)
    print("\nForecasting pipeline executed successfully.")
    print(f"Results saved to: {project_root / 'outputs'}")