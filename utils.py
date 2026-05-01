# File Management Libraries
import os
from itertools import product
from collections import defaultdict

# Finance Libraries
import yfinance as yf
import empyrical as emp

# Data Science Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.stats import skew, kurtosis, norm
from sklearn.metrics import mean_squared_error as rmse, mean_absolute_error as mae, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline


rp_pca_cols = [f'RPPCA_{i}' for i in range(1, 6)]
pca_cols = [f'PCA_{i}' for i in range(1, 6)]
all_factor_cols = rp_pca_cols + pca_cols

#--------------------------------------------------------------------------------------------------#

def import_all_data():
    factors_data = import_factors_data()
    spy_data = import_spy_data()

    all_data = factors_data.join(spy_data, how='inner')
    all_data['SPY'] = np.log(all_data['SPY']).diff().shift(-1)
    all_data = all_data.dropna()

    return all_data

def import_factors_data():
    factor_file_path = os.path.join(os.getcwd(), 'RP-PCA Factors.csv')
    factors_data = pd.read_csv(factor_file_path, encoding='latin1')

    # Make row 5 the column headers
    factors_data.columns = factors_data.iloc[5]

    # Drop rows 0 to 5 (inclusive)
    factors_data = factors_data.iloc[6:, :]

    # Remove unwanted columns
    factors_data = factors_data.drop(columns=factors_data.columns[factors_data.columns.isna()])

    # Make Date columns datetime type
    factors_data['date'] = pd.to_datetime(factors_data['date'], format='%b-%y')
    factors_data['date'] = factors_data['date'].apply(lambda x : x.replace(year=x.year - 100) if x.year > 2017 else x)

    # Make date column index (of type DateTime)
    factors_data = factors_data.set_index('date')
    factors_data.index = factors_data.index.to_period('M')

    # Fix data types of rest of columns
    factors_data = factors_data.astype('float64')

    # Negate Factors
    factors_data = -factors_data

    return factors_data

def import_spy_data():
    target_etfs = 'SPY'
    start_date, end_date = '1963-11-01', '2017-12-31'
    spy_data = yf.download([target_etfs], start=start_date, end=end_date, auto_adjust=True)
    spy_data = spy_data['Close'].resample('M').last()
    spy_data.index = spy_data.index.to_period('M')
    
    return spy_data

#----------------------------------------------------------------------------------------------#

def create_lagged_df(original_df, h):
    df = original_df.copy()
    df['SPY_new'] = df['SPY']
    for i in range(1,h):
        df['SPY_new'] += df['SPY'].shift(-i)
    
    df = df.dropna()
    return df

def create_train_predict_calendar(df, window_type, window_size):
    date_ranges = []
    index = df.index
    num_days = len(index)

    if window_type == 'fixed':
        # Rolling window: Use only recent window_size periods for training
        for i in range(window_size, num_days):
            train_start_date = index[i - window_size]
            train_end_date = index[i - 1]
            prediction_date = index[i]
            date_ranges.append([train_start_date, train_end_date, prediction_date])

    elif window_type == 'expanding':
        # Expanding window: Use all available historical data
        for i in range(window_size, num_days):
            train_start_date = index[0]
            train_end_date = index[i - 1]
            prediction_date = index[i]
            date_ranges.append([train_start_date, train_end_date, prediction_date])

    return date_ranges

def compute_oos_metrics(results_df):
    df = results_df.copy().dropna()

    actual = df['Actual SPY Return']
    pred_model = df['Predicted SPY Return']
    pred_hist = df['Historical Mean Prediction']

    # forecast errors
    err_model = actual - pred_model
    err_hist = actual - pred_hist

    # MSPE
    mspe_model = np.mean(err_model**2)
    mspe_hist = np.mean(err_hist**2)

    # Campbell-Thompson OOS R^2
    r2_os = 1 - mspe_model / mspe_hist

    # Clark-West adjusted series
    cw_series = (err_hist**2) - (err_model**2) + (pred_hist - pred_model)**2

    # one-sided Clark-West test: H1 = model beats benchmark
    cw_mean = cw_series.mean()
    cw_se = cw_series.std(ddof=1) / np.sqrt(len(cw_series))
    cw_tstat = cw_mean / cw_se if cw_se > 0 else np.nan
    cw_pvalue = 1 - stats.norm.cdf(cw_tstat) if np.isfinite(cw_tstat) else np.nan

    # success ratio
    success = np.mean(np.sign(pred_model) == np.sign(actual))

    return {
        'n_oos': len(df),
        'MSPE_model': mspe_model,
        'MSPE_hist_mean': mspe_hist,
        'R2_OS': r2_os,
        'CW_tstat': cw_tstat,
        'CW_pvalue_one_sided': cw_pvalue,
        'Success_Ratio': success,
        'SPY_Up_Proportion': np.mean(actual > 0)
    }

def run_OLS(X, y, add_constant, heteroskedasticity=True):
    output = {}

    model = sm.OLS(y, X)
    if add_constant:
        model = sm.OLS(y, sm.add_constant(X))

    if heteroskedasticity:
        result = model.fit()
    else:
        result = model.fit({})

    output["t-values"] = result.tvalues
    output["p-values"] = result.pvalues
    output["r-squared"] = result.rsquared

    return result, output