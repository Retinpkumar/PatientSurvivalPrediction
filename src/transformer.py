import pandas as pd
import numpy as np


def input_transformer():
    # reading data
    df = pd.read_csv('data/Xnum.csv')
    # log transformation
    df['gcs_verbal_apache'] = np.log1p(df['gcs_verbal_apache'])
    # Winsorizing
    q1 = df['gcs_verbal_apache'].quantile(0.25)
    q3 = df['gcs_verbal_apache'].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + (1.5 * iqr)
    lower = q1 - (1.5 * iqr)
    df['gcs_verbal_apache'] = np.where(df['gcs_verbal_apache'] > upper, upper, np.where(df['gcs_verbal_apache'] < lower, lower, df['gcs_verbal_apache']))
    return df
