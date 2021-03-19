# %%
# test maps version
import os

os.environ['CITYSEER_QUIET_MODE'] = '1'

from cityseer.util import mock, graphs
from src.explore.MMM.layercake_b import mmm_layercake_b

iters = 200
graph = mock.mock_graph()
graph = graphs.nX_simple_geoms(graph)
graph = graphs.nX_decompose(graph, 20)

layer_specs = {
    'cap_step': 0.5,
    'dist_threshold': 600,
    'pop_threshold': 30,
    'kill_threshold': 0.5,
    'explore_rate': 0.5
}

pop_map, landuse_maps, capacitance_maps = mmm_layercake_b(graph,
                                                          iters,
                                                          _layer_specs=layer_specs,
                                                          seed=False,
                                                          random_seed=0)

# %%
import networkx as nx
from src import phd_util
import matplotlib.pyplot as plt
from celluloid import Camera
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('cityseer', ['#64c1ff', '#d32f2f'])

phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(20, 20), facecolor='#2e2e2e')
camera = Camera(fig)

pos = {}
for n, d in graph.nodes(data=True):
    pos[n] = (d['x'], d['y'])

for i in range(iters):
    lus = landuse_maps[0][i]
    caps = capacitance_maps[0][i]
    mags = lus * caps

    nx.draw_networkx(graph,
                     pos,
                     ax=ax,
                     with_labels=False,
                     font_size=5,
                     font_color='w',
                     font_weight='bold',
                     node_color=mags,
                     node_size=mags * 150,
                     cmap=cmap,
                     node_shape='o',
                     edge_color='r',
                     width=1,
                     alpha=1)
    camera.snap()

ax.set_axis_off()
theme = 'test_map'
fig.suptitle(theme)

animation = camera.animate()
animation.save(f'./exploratory_plots/{theme}.mp4')
