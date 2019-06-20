"""
用于对ECG数据集进行预处理
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


def load_files(path, norm=True, save=False,
               suffix='txt', inx=-1, delimiter=','):
    """
    Args:
        path: 多个数据集所在文件夹路径
        norm: 是否规范化处理
        save: 是否存储为npz文件
        suffix: 数据集文件后缀
        inx: 标签所在列索引，0或者-1
        delimiter: 每一列数据分隔符
    """
    path = Path(path)
    xs = []
    ys = []
    if inx not in [0, -1]:
        raise ValueError('inx is 0/-1')
    for p in path.glob('*.'+suffix):
        data = np.loadtxt(p, delimiter=delimiter)
        if inx == 0:
            x, y = data[:, 1:], data[:, 0]
        elif inx == -1:
            x, y = data[:, 0:-1], data[:, -1]
        xs.append(x)
        ys.append(y)
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    if norm:
        scaler = StandardScaler()
        xs = scaler.fit_transform(xs)
        scaler = MinMaxScaler()
        xs = scaler.fit_transform(xs)
    if save:
        np.savez(
            path.joinpath(path.name),
            data=xs,
            label=ys,
        )
    return xs, ys


def load_npz(name='ECG200'):
    path = '../data/{0}/{0}.npz'.format(name)
    if not os.path.exists(path):
        path = './data/{0}/{0}.npz'.format(name)
    data = np.load(path)
    return data['data'], data['label']

if __name__ == "__main__":
    x, y = load_npz('FordA')
    # x, y = load_files('../data/ECG200/', inx=0, suffix='tsv', delimiter='\t')
    print(x.shape, y.shape)
    plt.figure()
    plt.plot(x[0, :])
    plt.show()
