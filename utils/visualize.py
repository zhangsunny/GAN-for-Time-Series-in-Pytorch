import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def trans_dataframe(data, name='gen'):
    x = np.array(data)
    n, m = x.shape
    y = np.zeros([np.prod(x.shape), 2])
    for i in range(n):
        y[i*m:i*m+m, 0] = np.arange(m).astype(np.int)
        y[i*m:i*m+m, 1] = x[i, :]
    df = pd.DataFrame(data=y, columns=['time_point', 'value'])
    df['type'] = name
    return df


def trans_dataframes(data: dict):
    df = []
    for k, v in data.items():
        df.append(trans_dataframe(data=v, name=k))
    d = pd.concat(df)
    return d


def plot_errorband(data: dict):
    plt.figure(figsize=(15, 5))
    df = trans_dataframes(data)
    sns.lineplot(x='time_point', y='value', hue='type', data=df)
    plt.show()
