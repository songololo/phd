import os

os.environ['CITYSEER_QUIET_MODE'] = '1'

import networkx as nx
import matplotlib.pyplot as plt

from src import util_funcs
from src.explore.toy_models import generate_graph_lattice

# %%
g = generate_graph_lattice()

pos = {}
for n, d in g.nodes(data=True):
    pos[n] = (d['x'], d['y'])

util_funcs.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))

nx.draw_networkx(g, pos=pos, ax=ax)

plt.show()
