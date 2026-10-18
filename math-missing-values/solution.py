import numpy as np


def missing_mask(x: np.ndarray) -> np.ndarray:
    """Missing numeric values are represented as np.nan (NumPy/pandas' standard convention).

    Returns a boolean array the same shape as x, True wherever a value is
    missing.
    """
    return np.isnan(x)


def missing_count_per_column(x: np.ndarray) -> np.ndarray:
    """x is a 2D array, (rows, columns).

    Returns a length-`num_columns` array: how many missing values each column
    has.
    """
    return np.isnan(x).sum(axis=0)


def missing_fraction_per_column(x: np.ndarray) -> np.ndarray:
    """Same as missing_count_per_column, but as a fraction of the total row

    count (0.0 to 1.0) rather than a raw count.
    """
    return np.isnan(x).mean(axis=0)


def impute_with_mean(x: np.ndarray) -> np.ndarray:
    """Replace every missing (NaN) value with its OWN COLUMN's mean, computed

    from that column's non-missing values only (np.nanmean ignores NaNs
    automatically, rather than propagating them into the mean itself). Must not
    mutate the input array, work on a copy.
    """
    out = x.copy()
    col_means = np.nanmean(out, axis=0)
    nan_mask = np.isnan(out)
    out[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return out


def impute_with_median(x: np.ndarray) -> np.ndarray:
    """Same idea as impute_with_mean, but replacing with each column's median

    (np.nanmedian) instead.
    """
    out = x.copy()
    col_medians = np.nanmedian(out, axis=0)
    nan_mask = np.isnan(out)
    out[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return out
