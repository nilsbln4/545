"""Return calculation from a price table.

Port of library/return_calculate.jl.
"""

import numpy as np
import pandas as pd


def return_calculate(prices: pd.DataFrame, method: str = "DISCRETE",
                      date_column: str = "date") -> pd.DataFrame:
    if date_column not in prices.columns:
        raise ValueError(f"date_column: {date_column} not in DataFrame: {list(prices.columns)}")

    value_cols = [c for c in prices.columns if c != date_column]
    p = prices[value_cols].to_numpy(dtype=float)
    p2 = p[1:, :] / p[:-1, :]

    method_u = method.upper()
    if method_u == "DISCRETE":
        p2 = p2 - 1.0
    elif method_u == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f'method: {method} must be in ("LOG","DISCRETE")')

    dates = prices[date_column].iloc[1:].reset_index(drop=True)
    returns = pd.DataFrame(p2, columns=value_cols)
    return pd.concat([dates, returns], axis=1)
