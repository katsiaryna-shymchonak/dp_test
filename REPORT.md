# Demand Planning (DP) Pipeline Report & Technical Audit

## 1. Data Quality & Anomaly Handling

| Anomaly / Defect | Detection Method & Count | Action Taken | Rationale |
| --- | --- | --- | --- |
| **Duplicate Sales Records** | Key duplication check on `[sku, location, period]` (3 duplicate rows) | Retained first occurrence via `drop_duplicates()` | Eliminates record multiplication during multi-table joins and prevents distorted demand aggregation. |
| **Negative Sales Quantities** | Boundary validation on `qty < 0` (1 record with `qty = -8`) | Clipped negative quantities to zero using `.clip(lower=0)` | Operational returns distort baseline demand and violate non-negativity model constraints. |
| **Ghost Stock Entries** | Cross-field filtering on `stock_end_qty == 0` and `days_out_of_stock == 0` (11 records) | Reassigned `days_out_of_stock = 30` for affected periods | Zero period-end inventory without logged out-of-stock days signals an inventory logging system error. |
| **Full-Month Stockouts (OOS)** | Identified 100% OOS periods (`days_out_of_stock >= 30`, masking 12,987 lost units) | Imputed unconstrained demand using SKU organic non-promo baseline medians | Linear proportional scaling yields zero when actual sales are zero; organic median imputation restores true unconstrained demand. |
| **Promotional Demand Spikes** | Comparative analysis on `discount_pct > 0` (8 promo periods with +134% to +410% uplift) | Pre-smoothed promotional periods to organic medians for baseline model fitting | Prevents promotional volume surges from corrupting baseline trend and 12-month seasonality estimation. |
| **Invalid Global IQR Bounds** | Global IQR analysis (lower bound -1552.12 units, 6.82% mislabeled as outliers) | Replaced global IQR trimming with targeted contextual cleaning | Scale variances across portfolio classes render global statistical bounds invalid, misflagging high-velocity A-class SKUs. |

---

## 2. Key Decision Log: Evolutionary Architecture & Stage Refinement

### Phase 1: Exploratory Data Analysis (EDA & Initial Hypotheses)

* **Baseline Candidates:** Formulated initial suite including Weighted Moving Average (WMA) for low-complexity benchmark, Holt-Winters for regular demand, Croston/TSB for zero-inflated series, and LightGBM with Tweedie Loss to handle skewed data without log-transform bias.

### Phase 2: Technical Strategy & Model Selection

* **Baseline Floor Setup:** Selected **Seasonal Naïve (12M Lag)** and **WMA-3** as the hard benchmark floor to verify whether complex algorithms deliver real uplift over simple historical lags or rolling averages.
* **Regular SKUs Engine:** Assigned **Holt-Winters Exponential Smoothing** as primary engine for 33 Smooth SKUs to capture trend and 12-month seasonality on pre-cleaned data. Added **Auto-SARIMA** $(1,1,1)(1,1,0)_{12}$ as a statistical benchmark to evaluate residual autocorrelation.
* **Non-Smooth SKUs Engine:** Selected **Teunter-Syntetos-Babai (TSB)** over classic Croston for 7 Lumpy/Erratic SKUs. TSB updates demand probability every period (including zeros), preventing severe over-forecasting after long stockouts.
* **ML Demotion:** Demoted **LightGBM (Tweedie)** to a secondary experiment, hypothesizing severe overfitting on small sample size ($N = 1,440$ rows: $40 \text{ SKUs} \times 36 \text{ months}$).

### Phase 3: Modeling & Backtest Adaptation

* **Ensemble Creation:** Out-of-sample backtesting revealed standalone Holt-Winters over-extrapolated linear trends. Introduced a **50/50 Ensemble Blend (HW/TSB + Seasonal Naïve)**, anchoring predictions to historical seasonal profiles and reducing portfolio WAPE from 31.82% to **27.03%**.
* **Time Constraint & Parameter Tuning Note:** Due to the strict 3-4 hour time budget, grid-search hyperparameter optimization ($\alpha, \beta, \gamma$ parameters for Holt-Winters/TSB) was omitted. Models run on default/estimated initialization parameters, leaving automated hyperparameter tuning as the primary immediate improvement.

---

## 3. SKU Demand Segmentation & Model Benchmarking

Holdout backtest conducted on out-of-sample history (33 months train / 3 months test evaluation).

| Model / Candidate | Target Segment / Role | Portfolio WAPE (%) | Portfolio MASE | Portfolio RMSE | Performance Verdict |
| --- | --- | --- | --- | --- | --- |
| **Ensemble (HW/TSB + SNaive)** | **Production Champion** | **27.03%** | **1.53** | 2501.20 | **Winner:** 50/50 blending mitigates trend overshoot with historical seasonal baseline. |
| **Seasonal Naïve (12M Lag)** | Baseline Benchmark | 28.70% | 1.63 | 2383.75 | Effectively captures annual peaks, but vulnerable to single-year historical anomalies. |
| **Weighted Moving Average (WMA-3)** | Baseline Benchmark | 30.19% | 1.59 | **1874.00** | **Lowest Variance:** Smooths outliers well but completely misses annual seasonality. |
| **Primary Champion (HW / TSB)** | Statistical Engine | 31.82% | 1.70 | 2675.04 | Dynamic engine (33 Smooth, 7 Lumpy/Erratic); over-extrapolates localized trend. |
| **Auto-SARIMA $(1,1,1)(1,1,0)_{12}$** | Statistical Benchmark | 31.82% | 1.70 | 2674.73 | Benchmark matches HW performance, validating model convergence and parameter selection. |
| **LightGBM (Tweedie Loss)** | Secondary ML Fallback | 66.09% | 3.56 | 3514.99 | **Rejected:** Overfits on small sample size ($N=1,440$) and produces flat predictions. |

---

## 4. Key Business Questions for Domain Stakeholders

1. **What are the asymmetric holding vs. stockout costs per product category?**
* *Impact:* Models currently optimize for symmetric loss. Understanding margin structures allows tuning asymmetric quantile loss functions to prevent high-cost stockouts on A-class SKUs.


2. **What post-promotional demand effects (cannibalization vs. dip-after-promo) are typical?**
* *Impact:* Clarifying whether promotional spikes borrow demand from adjacent months allows modeling lag effects without misinterpreting post-promo dips as structural downward trends.


3. **What are the true replenishment lead times for intermittent (Lumpy) SKUs?**
* *Impact:* For long replenishment lead times ($>2$ months), a 3-month point forecast is insufficient; supply chain teams require cumulative horizon quantile distribution forecasts to set safety stock.



---

## 5. Model Weaknesses & Specific SKU Failures

* **SKU_0002 (Lumpy Demand Segment):** TSB outputs a flat expected demand rate ($\sim 25.65$ units/month). The model continuously over-forecasts during zero-demand months and under-forecasts during sudden order bursts.
* **SKU_0038 (Erratic Demand Segment):** High demand variance ($\text{CV}^2 \ge 0.49$) causes seasonal extrapolation in Holt-Winters to amplify random noise, producing wider confidence interval errors during peak months.

---

## 6. Scale & Production Operations (500,000 SKUs)

* **Architecture Modifications:**
* Transition feature engineering pipelines from `pandas` to `Polars` or `PyArrow` for vectorized memory efficiency.
* Implement distributed chunk processing across SKU clusters using `Ray` or `Dask`.
* Replace Python loop statistical routines with C-compiled batch implementations or global LightGBM with recursive lag updates once historical volume exceeds $10^6$ rows.


* **Proactive Drift Detection:**
* **Tracking Signal Monitoring:** Calculate $\text{TS} = \frac{\sum (y_t - \hat{y}_t)}{\text{MAD}_t}$ per SKU. Values outside $[-4, 4]$ trigger automated alerts for bias drift prior to business disruption.
* **Rolling Out-of-Sample WAPE Alerts:** Automatically evaluate 1-month-ahead forecast errors against 12-month historical baselines. A relative degradation $>15\%$ triggers automatic model re-segmentation and re-tuning.