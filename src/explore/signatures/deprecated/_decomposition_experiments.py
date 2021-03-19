# %%

import matplotlib.pyplot as plt
import numpy as np
from src import phd_util
from sklearn.preprocessing import minmax_scale, robust_scale

X = np.array([
    [1, 1, 1],
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0]])
y = np.array([3, 2, 1, 0])


def X_smushed(_X):
    X = np.copy(_X)
    X = minmax_scale(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    assert X.ndim == 2
    dims = X.shape[1]
    for d in range(dims, 0, -1):
        X[:, d - 1] = np.product(X[:, :d], axis=1)
    X = robust_scale(X)
    return X


print(X_smushed(X))


def X_diffed(_X):
    X = np.copy(_X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    assert X.ndim == 2
    dims = X.shape[1]
    for d in range(1, dims):
        X[:, d] = X[:, d] - np.sum(X[:, :d], axis=1)
    X = robust_scale(X)
    return X


print(X_diffed(X))


def plotter(red, title):
    phd_util.plt_setup()
    fig, ax = plt.subplots(1, 1)
    for n, y_label in enumerate(y):
        ax.scatter(red[n, 0], red[n, 1])
        ax.text(red[n, 0], red[n, 1], y_label)
    plt.suptitle(title)
    plt.show()


# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit_transform(X)
print(pca.round(2))
plotter(pca, 'pca')

# %%
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2).fit_transform(X)
print(svd.round(2))
plotter(svd, 'svd')

# %%
from sklearn.decomposition import DictionaryLearning

dictLearn = DictionaryLearning(n_components=2).fit_transform(X)
print(dictLearn.round(2))
plotter(dictLearn, 'dictLearn')

# %%
from sklearn.decomposition import FactorAnalysis

factor = FactorAnalysis(n_components=2).fit_transform(X)
print(factor.round(2))
plotter(factor, 'factor')

# %%
from sklearn.decomposition import FastICA

fast = FastICA(n_components=2).fit_transform(X)
print(fast.round(2))
plotter(fast, 'fast')

# %%
from sklearn.decomposition import NMF

nmf = NMF(n_components=2).fit_transform(X)
print(nmf.round(2))
plotter(nmf, 'nmf')

# %%
from sklearn.decomposition import KernelPCA

kPCA = KernelPCA(n_components=2).fit_transform(X)
print(kPCA.round(2))
plotter(kPCA, 'kPCA')

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis().fit_transform(X, y)
print(lda.round(2))
plotter(lda, 'lda')

# %%
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis().fit(X, y)
y_hat = qda.predict(X)
print(y_hat.round(2))
