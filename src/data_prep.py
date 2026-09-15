import numpy as np
import pandas as pd


def clean_sales_records(sales: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate sales records and clips negative sales entries to zero."""
    # Remove exact structural duplicates
    df_sales = sales.drop_duplicates(
        subset=['sku', 'location', 'period']
    ).copy()

    # Correct negative sales entries to zero
    df_sales['qty'] = df_sales['qty'].clip(lower=0)
    return df_sales


def fix_ghost_stock(stock: pd.DataFrame) -> pd.DataFrame:
    """Corrects ghost stock where zero stock is recorded with zero OOS days."""
    df_stock = stock.copy()
    ghost_mask = (df_stock['stock_end_qty'] == 0) & (
        df_stock['days_out_of_stock'] == 0
    )
    df_stock.loc[ghost_mask, 'days_out_of_stock'] = 30
    return df_stock


def map_predecessors(df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Maps predecessor items to handle cold starts."""
    if 'predecessor_sku' not in items.columns:
        return df

    mapping = (
        items.dropna(subset=['predecessor_sku'])
        .set_index('predecessor_sku')['sku']
        .to_dict()
    )
    df_mapped = df.copy()
    df_mapped['sku'] = df_mapped['sku'].replace(mapping)
    return df_mapped


def restore_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Restores true demand for out-of-stock periods securely using organic baseline median."""
    df_res = df.copy()

    # Calculate clean organic baseline median per SKU excluding promo and stockout periods
    clean_mask = (df_res['days_out_of_stock'] == 0) & (
        df_res['discount_pct'] == 0
    )
    sku_organic_medians = df_res[clean_mask].groupby('sku')['qty'].median()

    # Restore true demand for out-of-stock periods securely
    conditions = [
        df_res['days_out_of_stock'] == 0,
        (df_res['days_out_of_stock'] > 0) & (df_res['days_out_of_stock'] < 30),
    ]
    choices = [
        df_res['qty'],
        (df_res['qty'] / (30 - df_res['days_out_of_stock'])) * 30,
    ]

    df_res['qty_restored'] = np.select(
        conditions,
        choices,
        default=df_res['sku'].map(sku_organic_medians),
    )
    return df_res


def clean_data(
    sales: pd.DataFrame,
    stock: pd.DataFrame,
    items: pd.DataFrame,
    promos: pd.DataFrame = None,
) -> pd.DataFrame:
    """Main orchestration pipeline for data cleaning and demand restoration."""
    sales_clean = clean_sales_records(sales)
    stock_clean = fix_ghost_stock(stock)

    df = pd.merge(
        sales_clean, stock_clean, on=['sku', 'location', 'period'], how='left'
    )
    df['days_out_of_stock'] = df['days_out_of_stock'].fillna(0)
    df['stock_end_qty'] = df['stock_end_qty'].fillna(0)

    if promos is not None and not promos.empty:
        df = pd.merge(df, promos, on=['sku', 'period'], how='left')
        df['discount_pct'] = df['discount_pct'].fillna(0)
    else:
        df['discount_pct'] = 0

    # Map predecessor items to handle cold starts
    df = map_predecessors(df, items)

    # Restore true demand for out-of-stock periods securely
    df = restore_demand(df)

    # Aggregate after remapping keys
    df_final = (
        df.groupby(['sku', 'location', 'period'], as_index=False)
        .agg({
            'qty_restored': 'sum',
            'days_out_of_stock': 'max',
            'discount_pct': 'max',
        })
        .rename(columns={'qty_restored': 'qty'})
    )

    return df_final