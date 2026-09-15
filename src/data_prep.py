import pandas as pd
import numpy as np


def clean_data(sales: pd.DataFrame, stock: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    # remove exact structural duplicates
    sales = sales.drop_duplicates(subset=['sku', 'location', 'period'])

    # correct negative sales entries to zero
    sales.loc[sales['qty'] < 0, 'qty'] = 0

    df = pd.merge(sales, stock, on=['sku', 'location', 'period'], how='left')
    df['days_out_of_stock'] = df['days_out_of_stock'].fillna(0)

    # restore true demand for out-of-stock periods securely
    df['qty'] = np.where(
        df['days_out_of_stock'] < 30,
        df['qty'] / (30 - df['days_out_of_stock']) * 30,
        df['qty']
    )

    # map predecessor items to handle cold starts
    if 'predecessor_sku' in items.columns:
        mapping = items.dropna(subset=['predecessor_sku']).set_index('predecessor_sku')['sku'].to_dict()
        df['sku'] = df['sku'].replace(mapping)

    # aggregate after remapping keys
    df = df.groupby(['sku', 'location', 'period'], as_index=False)['qty'].sum()

    return df