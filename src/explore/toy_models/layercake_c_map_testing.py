# %%
# test maps version
import os

os.environ['CITYSEER_QUIET_MODE'] = '1'

from cityseer.util import mock, graphs
from src.explore.toy_models.layercake_c import mmm_layercake_c

iters = 250
graph = mock.mock_graph()
graph = graphs.nX_simple_geoms(graph)
graph = graphs.nX_decompose(graph, 20)

layer_specs = {
    'cap_step': 0.1,
    'dist_threshold': 1200,
    'pop_threshold': 100,
    'explore_rate': 0.5,
    'comp_rate': 0.1
}

pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_c(graph,
                                                                     iters,
                                                                     _layer_specs=layer_specs,
                                                                     random_seed=0)

# %%
from tqdm import tqdm
import numpy as np
from src import util_funcs
import matplotlib.pyplot as plt
from celluloid import Camera
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

cmap = LinearSegmentedColormap.from_list('cityseer', ['#64c1ff', '#d32f2f'])

util_funcs.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
camera = Camera(fig)

pos = {}
for n, d in graph.nodes(data=True):
    pos[n] = (d['x'], d['y'])
xs = np.array([v[0] for v in pos.values()])
ys = np.array([v[1] for v in pos.values()])

lines = []
for s, e in graph.edges():
    s_x = graph.nodes[s]['x']
    s_y = graph.nodes[s]['y']
    e_x = graph.nodes[e]['x']
    e_y = graph.nodes[e]['y']
    lines.append([[s_x, s_y], [e_x, e_y]])

for i in tqdm(range(iters)):
    pop = pop_map[i]
    pop /= np.nanmax(pop)

    lus = landuse_maps[0][i]
    lus /= np.nanmax(lus)

    caps = capacitance_maps[0][i]
    caps /= np.nanmax(caps)

    flows = flow_maps[0][i]
    flows /= np.nanmax(flows)

    ax.set_facecolor('#2e2e2e')
    ax.add_collection(LineCollection(lines, color='w', linewidth=0.5))
    ax.scatter(xs, ys, s=flows * 1000, facecolors='#2e2e2e', edgecolors='#2e7d32', linewidths=1, alpha=1)
    ax.scatter(xs, ys, s=caps * 250, facecolors='#d32f2f', edgecolors='#9a0007', linewidths=1, alpha=1)
    ax.scatter(xs, ys, s=lus * 100, facecolors='#0091ea', edgecolors='#0064b7', linewidths=1, alpha=1)
    ax.scatter(xs, ys, s=pop * 10, c='#ffffff', alpha=1)
    camera.snap()

ax.set_axis_off()
theme = 'test_map'
fig.suptitle(theme)

animation = camera.animate()

explore_rate = str(layer_specs["explore_rate"])
explore_rate = explore_rate.replace('.', '_')

comp_rate = str(layer_specs["comp_rate"])
comp_rate = comp_rate.replace('.', '_')

animation.save(f'./src/explore/H_mmm/exploratory_plots/{theme}_exp_{explore_rate}_comp_{comp_rate}.mp4')
