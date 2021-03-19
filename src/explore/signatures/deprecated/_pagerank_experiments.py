"""

"""

import asyncio

import matplotlib.pyplot as plt
# %%
import numpy as np
from src import phd_util
import scipy
import scipy.sparse.linalg as scipy_linalg
from cityseer.util import graphs
from networkx.algorithms.centrality.flow_matrix import *
from process.loaders import postGIS_to_networkX
from scipy import sparse
from sklearn.preprocessing import minmax_scale

db_config = {
    'host': 'localhost',
    'port': 5433,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}

boundary_table = 'analysis.city_boundaries_150'

nodes_table = 'analysis.roadnodes_20'
links_table = 'analysis.roadlinks_20'

distances = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]

# %%
G = asyncio.run(postGIS_to_networkX(db_config, nodes_table, links_table, boundary_table, 1))
G = graphs.nX_auto_edge_params(G)  # generate default lengths and impedances based on geom lengths

# %%
connected = nx.connected_components(G)
largest = max(connected, key=len)
G_pruned = G.subgraph(largest).copy()


# %%
def plotter(label):
    phd_util.plt_setup()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(xs,
               ys,
               c=dt_arr,
               s=2,
               # alpha=1,
               # marker='.',
               # linewidths=0,
               # edgecolors='none',
               cmap='plasma')
    x_center = np.nanmedian(xs)
    y_center = np.nanmedian(ys)
    offset = 5000
    ax.set_xlim(left=x_center - offset, right=x_center + offset)
    ax.set_ylim(bottom=y_center - offset, top=y_center + offset)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)

    plt.suptitle(f'PageRank experiment')

    plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/{label}.png', dpi=300)
    plt.show()


# %%
### FAST
from networkx.algorithms.centrality import eigenvector_centrality_numpy

eig_cent = eigenvector_centrality_numpy(G_pruned, max_iter=1000, tol=1.0e-6)  # weight='weight',
xs = []
ys = []
dt = []
for node, data in G_pruned.nodes(data=True):
    xs.append(data['x'])
    ys.append(data['y'])
    dt.append(eig_cent[node])
dt_arr = np.array(dt)
dt_arr = np.clip(dt_arr, np.percentile(dt_arr, 1), np.percentile(dt_arr, 99))
dt_arr = minmax_scale(dt_arr)
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.hist(dt_arr, bins=100)
plt.show()

# %%
### CRASHES

# katz_cent = katz_centrality_numpy(G_pruned)  # weight='weight',
xs = []
ys = []
dt = []
for node, data in G_pruned.nodes(data=True):
    xs.append(data['x'])
    ys.append(data['y'])
    dt.append(katz_cent[node])
dt_arr = np.array(dt)
dt_arr = np.clip(dt_arr, np.percentile(dt_arr, 1), np.percentile(dt_arr, 99))
dt_arr = minmax_scale(dt_arr)
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.hist(dt_arr, bins=100)
plt.show()

# %%
#### SLOW

# curr_cent = current_flow_closeness_centrality(G_pruned)
xs = []
ys = []
dt = []
for node, data in G_pruned.nodes(data=True):
    xs.append(data['x'])
    ys.append(data['y'])
    dt.append(curr_cent[node])
dt_arr = np.array(dt)
dt_arr = np.clip(dt_arr, np.percentile(dt_arr, 1), np.percentile(dt_arr, 99))
dt_arr = minmax_scale(dt_arr)
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.hist(dt_arr, bins=100)
plt.show()

# %%
### CRASHES

# curr_betw_cent = approximate_current_flow_betweenness_centrality(G_pruned, solver='full')
xs = []
ys = []
dt = []
for node, data in G_pruned.nodes(data=True):
    xs.append(data['x'])
    ys.append(data['y'])
    dt.append(curr_betw_cent[node])
dt_arr = np.array(dt)
dt_arr = np.clip(dt_arr, np.percentile(dt_arr, 1), np.percentile(dt_arr, 99))
dt_arr = minmax_scale(dt_arr)
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.hist(dt_arr, bins=100)
plt.show()

# %%
### CRASHES
from networkx.algorithms.centrality import second_order_centrality

second_order_cent = second_order_centrality(G_pruned)
xs = []
ys = []
dt = []
for node, data in G_pruned.nodes(data=True):
    xs.append(data['x'])
    ys.append(data['y'])
    dt.append(second_order_cent[node])
dt_arr = np.array(dt)
dt_arr = np.clip(dt_arr, np.percentile(dt_arr, 1), np.percentile(dt_arr, 99))
dt_arr = minmax_scale(dt_arr)
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.hist(dt_arr, bins=100)
plt.show()

# %%
G_directed = G_pruned.to_directed()
for node in G_directed.nodes():
    out = G_directed.out_degree(node)
    for s, e in G_directed.out_edges(node):
        G_pruned[s][e]['weight'] = 1 / out

# %%
G_sparse = nx.to_scipy_sparse_matrix(G_directed, nodelist=list(G_pruned), weight='weight', dtype=float)

# %%
steps = 1000
scores = np.ones(G_sparse.shape[0])
aggs = np.zeros(G_sparse.shape[0])
M = G_sparse.copy()
for i in range(steps):
    # dot
    scores = M.T @ scores
    # add random factor
    # scores += np.random.randn(len(scores)) * 2
    # scores = np.clip(scores, 0, scores.max())
    # normalise
    scores = np.linalg.norm(scores)
    aggs += scores
aggs /= steps
xs = []
ys = []
dt = []
for n, (node, data) in enumerate(G_directed.nodes(data=True)):
    xs.append(data['x'])
    ys.append(data['y'])
    dt.append(aggs[n])
dt_arr = np.array(dt)
dt_arr = np.clip(dt_arr, np.percentile(dt_arr, 0.5), np.percentile(dt_arr, 99.5))
dt_arr = minmax_scale(dt_arr)
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.hist(dt_arr, bins=100)
plt.show()

# %%
plotter(f'stepper_{steps}')

# %%
# https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/centrality/eigenvector.html
# don't know how to apply damping here - creates artefacts

eigen_values, eigen_vectors = sparse.linalg.eigs(G_sparse.T, k=1, which='LM')
eigen_values = np.real(eigen_values)
eigen_vectors = np.real(eigen_vectors)

primary_e_vec = eigen_vectors.flatten().real
norm = scipy.sign(primary_e_vec.sum()) * scipy.linalg.norm(primary_e_vec)
state = primary_e_vec / norm

# %%
# https://github.com/elegant-scipy/elegant-scipy/blob/master/markdown/ch6.markdown
from scipy.sparse.linalg import spsolve

n = G_sparse.shape[0]
# np.seterr(divide='ignore')
# degrees = np.ravel(G_sparse.sum(axis=1))
# Deginv = sparse.diags(1 / degrees).tocsr()
# G_sparse = (Deginv @ G_sparse).T

damping = 0.05
beta = 1 - damping

I = sparse.eye(n, format='csc')

pagerank = spsolve(I - damping * G_sparse.T, np.full(n, beta / n))
print(pagerank)

# %%
print(state)
xs = []
ys = []
dt = []
for n, (node, data) in enumerate(G_directed.nodes(data=True)):
    xs.append(data['x'])
    ys.append(data['y'])
    dt.append(state[n])

# %%
# https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank_scipy

from networkx.algorithms.link_analysis import pagerank_scipy

# converts to sparse internally
pagerank = pagerank_scipy(G_directed,
                          alpha=0.01,
                          personalization=None,
                          max_iter=100,
                          tol=1.0e-6,
                          weight='weight',
                          dangling=None)

xs = []
ys = []
dt = []
for node, data in G_directed.nodes(data=True):
    xs.append(data['x'])
    ys.append(data['y'])
    dt.append(pagerank[node])

# %%
dt_arr = np.array(dt)
dt_arr = np.clip(dt_arr, np.percentile(dt_arr, 1), np.percentile(dt_arr, 99))
dt_arr = minmax_scale(dt_arr)

# %%
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.hist(dt_arr, bins=100)
plt.show()
