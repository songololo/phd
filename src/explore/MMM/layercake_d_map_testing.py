# %%
# test maps version
import os

os.environ['CITYSEER_QUIET_MODE'] = '1'

from cityseer.util import mock, graphs
from src.explore.MMM.layercake_d import mmm_layercake_d
from matplotlib import animation

import numpy as np
from src import util_funcs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

# %%
iters = 200
graph = mock.mock_graph()
graph = graphs.nX_simple_geoms(graph)
graph = graphs.nX_decompose(graph, 20)

layer_specs = {
    'cap_step': 0.1,
    'dist_threshold': 1200,
    'pop_threshold': 100,
    'explore_rate': 0.5,
    'comp_rate': 0.1,
    'echo_rate': 0.1
}

pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_d(graph,
                                                                     iters,
                                                                     _layer_specs=layer_specs,
                                                                     random_seed=0)

# %%
cmap = LinearSegmentedColormap.from_list('cityseer', ['#64c1ff', '#d32f2f'])

util_funcs.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))

pos = {}
for n, d in graph.nodes(data=True):
    pos[n] = (d['x'], d['y'])
xs = np.array([v[0] for v in pos.values()])
ys = np.array([v[1] for v in pos.values()])

ax.set_axis_off()
ax.set_xlim(np.min(xs), np.max(xs))
ax.set_ylim(np.min(ys), np.max(ys))

lines = []
for s, e in graph.edges():
    s_x = graph.nodes[s]['x']
    s_y = graph.nodes[s]['y']
    e_x = graph.nodes[e]['x']
    e_y = graph.nodes[e]['y']
    lines.append([[s_x, s_y], [e_x, e_y]])

empty_size_arr = np.full(xs.shape[0], 0.0)
flows_scatter = ax.scatter(xs, ys, s=empty_size_arr, facecolors='#2e2e2e', edgecolors='#2e7d32', linewidths=1, alpha=1)
caps_scatter = ax.scatter(xs, ys, s=empty_size_arr, facecolors='#d32f2f', edgecolors='#9a0007', linewidths=1, alpha=1)
lus_scatter = ax.scatter(xs, ys, s=empty_size_arr, facecolors='#0091ea', edgecolors='#0064b7', linewidths=1, alpha=1)
pop_scatter = ax.scatter(xs, ys, s=empty_size_arr, c='#ffffff', alpha=1)

explore_rate = str(layer_specs["explore_rate"])
explore_rate = explore_rate.replace('.', '_')

comp_rate = str(layer_specs["comp_rate"])
comp_rate = comp_rate.replace('.', '_')

theme = f'test_er{explore_rate}_cr{comp_rate}'
fig.suptitle(theme)
fp = f'./src/explore/MMM/exploratory_plots/{theme}.mp4'


def init_func():
    ax.set_facecolor('#2e2e2e')
    ax.add_collection(LineCollection(lines, color='w', linewidth=0.5))

    # return iterable of artists to be redrawn - ignored if no blit
    return pop_scatter, lus_scatter, caps_scatter, flows_scatter


def animate_func(i):
    '''
    "set_array" for colour
    "set_offsets" for x, y
    "set_sizes" for size
    '''
    pop = pop_map[i]
    pop /= np.nanmax(pop)
    pop_scatter.set_sizes(pop * 10)

    lus = landuse_maps[0][i]
    lus /= np.nanmax(lus)
    lus_scatter.set_sizes(lus * 100)

    caps = capacitance_maps[0][i]
    caps /= np.nanmax(caps)
    caps_scatter.set_sizes(caps * 250)

    flows = flow_maps[0][i]
    flows /= np.nanmax(flows)
    flows_scatter.set_sizes(flows * 1000)

    # return iterable of all modified or created artists - ignored if no blit
    return pop_scatter, lus_scatter, caps_scatter, flows_scatter


# for "frames" argument, int is piped to a range operator
anim = animation.FuncAnimation(fig,
                               func=animate_func,
                               frames=iters,
                               init_func=init_func,
                               interval=200,  # delay between frames in milliseconds - default 200
                               blit=True)
ff_writer = animation.FFMpegWriter(fps=24,
                                   codec='h264',
                                   bitrate=-1,  # auto
                                   metadata={
                                       'title': theme,
                                       'artist': 'Gareth Simons',
                                       'copyright': 'Gareth Simons'
                                   })
# %%
anim.save(fp, writer=ff_writer, dpi=218)  # iMac 5k
