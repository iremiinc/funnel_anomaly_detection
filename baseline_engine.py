import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

def calculate_mad_baseline(series: pd.Series, window: int = 24 * 7) -> tuple:
    """
    1. Robust Statistical Method: Moving Median & MAD
    """
    rolling_median = series.rolling(window=window, min_periods=24).median()
    
    def mad_func(x):
        med = np.median(x)
        return np.median(np.abs(x - med))
    
    rolling_mad = series.rolling(window=window, min_periods=24).apply(mad_func, raw=True)
    
    mad_std = 1.4826 * rolling_mad
    upper_bound = rolling_median + (3 * mad_std)
    lower_bound = np.maximum(0, rolling_median - (3 * mad_std))
    
    return rolling_median.bfill(), lower_bound.bfill(), upper_bound.bfill()


def calculate_stl_baseline(series: pd.Series, period: int = 24) -> tuple:
    """
    2. Seasonal Model: STL Decomposition
    """
    clean_series = series.interpolate(method='linear').bfill()
    
    stl = STL(clean_series, period=period, robust=True)
    res = stl.fit()
    
    expected = res.trend + res.seasonal
    residual = res.resid
    
    resid_std = np.std(residual)
    upper_bound = expected + (2.5 * resid_std)
    lower_bound = np.maximum(0, expected - (2.5 * resid_std))
    
    return expected, lower_bound, upper_bound


def calculate_ml_baseline(df: pd.DataFrame, target_col: str = "cr_payment_submitted_to_purchase_completed") -> tuple:
    """
    3. Forecasting Model: Scikit-Learn HistGradientBoostingRegressor

    """
    df_ml = df.copy()
    
    # Time Features
    df_ml['hour'] = df_ml['eventTime'].dt.hour
    df_ml['dayofweek'] = df_ml['eventTime'].dt.dayofweek
    df_ml['is_weekend'] = df_ml['dayofweek'].isin([5, 6]).astype(int)
    
    # Lag Features
    df_ml['lag_24h'] = df_ml[target_col].shift(24)
    df_ml['lag_168h'] = df_ml[target_col].shift(168)
    
    features = ['hour', 'dayofweek', 'is_weekend', 'lag_24h', 'lag_168h']
    
    train_mask = df_ml['lag_168h'].notna()
    
    X = df_ml.loc[train_mask, features]
    y = df_ml.loc[train_mask, target_col]
    
    # Scikit-learn Gradient Boosting Model
    model = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X, y)
    
    df_ml['expected_ml'] = np.nan
    df_ml.loc[train_mask, 'expected_ml'] = model.predict(X)
    
    df_ml['expected_ml'] = df_ml['expected_ml'].fillna(df_ml[target_col].median())
    
    residuals = y - model.predict(X)
    std_res = np.std(residuals)
    
    lower_bound = np.maximum(0, df_ml['expected_ml'] - (2.5 * std_res))
    upper_bound = df_ml['expected_ml'] + (2.5 * std_res)
    
    return df_ml['expected_ml'], lower_bound, upper_bound


if __name__ == "__main__":
    print("Overall Funnel Metrikleri Yükleniyor...")
    df_overall = pd.read_parquet("funnel_metrics_overall.parquet")
    df_overall['eventTime'] = pd.to_datetime(df_overall['eventTime'])
    df_overall = df_overall.sort_values('eventTime').reset_index(drop=True)
    
    target_metric = "cr_payment_submitted_to_purchase_completed"
    
    print(f"'{target_metric}' için Baseline Modelleri Çalıştırılıyor...")
    
    # 1. MAD Baseline
    mad_exp, mad_low, mad_high = calculate_mad_baseline(df_overall[target_metric])
    df_overall['mad_expected'] = mad_exp
    df_overall['mad_lower_bound'] = mad_low
    df_overall['mad_upper_bound'] = mad_high
    
    # 2. STL Baseline
    stl_exp, stl_low, stl_high = calculate_stl_baseline(df_overall[target_metric])
    df_overall['stl_expected'] = stl_exp
    df_overall['stl_lower_bound'] = stl_low
    df_overall['stl_upper_bound'] = stl_high
    
    # 3. Scikit-Learn ML Baseline
    ml_exp, ml_low, ml_high = calculate_ml_baseline(df_overall, target_col=target_metric)
    df_overall['ml_expected'] = ml_exp
    df_overall['ml_lower_bound'] = ml_low
    df_overall['ml_upper_bound'] = ml_high

    # Hibrit baseline
    df_overall['baseline_expected'] = (df_overall['stl_expected'] + df_overall['ml_expected']) / 2.0
    df_overall['baseline_lower_bound'] = np.minimum(df_overall['stl_lower_bound'], df_overall['ml_lower_bound'])
    df_overall['baseline_upper_bound'] = np.maximum(df_overall['stl_upper_bound'], df_overall['ml_upper_bound'])
    
    df_overall.to_parquet("funnel_metrics_with_baseline.parquet", index=False)
    
    print("Baseline hesaplaması tamamlandı ve 'funnel_metrics_with_baseline.parquet' kaydedildi!")