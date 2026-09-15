# Demand Planning (DP) Forecasting Pipeline

A modular demand forecasting pipeline designed for retail networks. The pipeline performs automated cleaning of historical sales records, fixes data defects (ghost stock, OOS stockouts), dynamically segments the product portfolio using the ADI / $\text{CV}^2$ matrix, logs model routing selections, and generates a 3-month out-of-sample forecast using a 50/50 statistical ensemble (Holt-Winters / TSB + Seasonal Naïve).

## Single-Command Quickstart

Ensure you have **Python 3.10+** installed. Run the following single command from the project root to install dependencies, run the test suite, and execute the end-to-end forecasting pipeline:

```bash
pip install -r requirements.txt; python -m pytest tests/; python src/pipeline.py