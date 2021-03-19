# %%
'''
STEP 3: plot animations of MMM in action across three scenarios

Uses predictive eating establishments model as a fitness function for CA simulation
'''
import os

os.environ['CITYSEER_QUIET_MODE'] = '1'

import pathlib
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from src import phd_util
from src.explore.predictions import step_C1_graphs
from src.explore.predictions.step_C2_MMM import mmm_layercake_phd

#  %%
'''
Three different landuses - compete with like-for-like vs. synergy with other landuses
'''
# '#9a0007'
# '#0064b7'
# '#0091ea'
# '#d32f2f'
layer_specs = [
    # the bank
    {
        'colour': [184 / 255, 42 / 255, 158 / 255, 1],
        'cap_step': 0.1,  # spatial lag between increase / decrease in location viability
        'dist_threshold': 1600,  # how far pedestrians are prepared to walk
        'pop_threshold': 100,  # the threshold for location success
        'to_x': 500
    },
    # the retail store
    {
        'colour': [252 / 255, 74 / 255, 43 / 255, 1],
        'cap_step': 0.2,  # spatial lag between increase / decrease in location viability
        'dist_threshold': 800,  # how far pedestrians are prepared to walk
        'pop_threshold': 30,  # the threshold for location success
        'to_x': 800
    },
    # the coffee shop
    {
        'colour': [0 / 255, 155 / 255, 176 / 255, 1],
        'cap_step': 0.4,  # spatial lag between increase / decrease in location viability
        'dist_threshold': 400,  # how far pedestrians are prepared to walk
        'pop_threshold': 10,  # the threshold for location success
        'to_x': 1150
    }
]

#  %% run models
'''
Three different graphs with distinctly different network topologies
'''
iters = 2000
plot_iters = 500
seed = 123
echo_rate = 0.25
echo_distance = 400
competition_factor = 1.1
death_rate = 0.0001
flow_jitter = 0.2
cap_jitter = 0.1

theme = 'stepped_caps'
title = f'{theme}_iter{iters}_seed{seed}'
title += f'_er{echo_rate}_ed{echo_distance}_cf{competition_factor}_dr{death_rate}_fj{flow_jitter}_cj{cap_jitter}'

#  %%
graph_a = step_C1_graphs.york_burb()
graph_b = step_C1_graphs.grid_ville()
graph_c = step_C1_graphs.suburb()

# create, else load data
dir_path = pathlib.Path('./temp_data/sim_data/')
if not dir_path.exists():
    dir_path.mkdir(exist_ok=True, parents=True)
if not pathlib.Path(dir_path / 'anim_a' / f'{title}_pop_map.npy').exists():
    anim_a = mmm_layercake_phd(graph_a,
                               iters,
                               echo_rate,
                               echo_distance,
                               competition_factor,
                               death_rate,
                               flow_jitter,
                               cap_jitter,
                               _layer_specs=layer_specs,
                               random_seed=seed)
    anim_b = mmm_layercake_phd(graph_b,
                               iters,
                               echo_rate,
                               echo_distance,
                               competition_factor,
                               death_rate,
                               flow_jitter,
                               cap_jitter,
                               _layer_specs=layer_specs,
                               random_seed=seed)
    anim_c = mmm_layercake_phd(graph_c,
                               iters,
                               echo_rate,
                               echo_distance,
                               competition_factor,
                               death_rate,
                               flow_jitter,
                               cap_jitter,
                               _layer_specs=layer_specs,
                               random_seed=seed)
    for anim, anim_key in zip([anim_a, anim_b, anim_c], ['a', 'b', 'c']):
        for data, data_key in zip(anim,
                                  ['pop_map', 'landuse_maps', 'capacitance_maps', 'flow_map']):
            dir_sub_path = pathlib.Path(dir_path / f'anim_{anim_key}')
            if not dir_sub_path.exists():
                dir_sub_path.mkdir(exist_ok=True, parents=True)
            np.save(str(dir_sub_path / f'{title}_{data_key}.npy'), data)
# otherwise load
else:
    anim_a = []
    anim_b = []
    anim_c = []
    for anim, anim_key in zip([anim_a, anim_b, anim_c],
                              ['a', 'b', 'c']):
        dir_sub_path = pathlib.Path(dir_path / f'anim_{anim_key}')
        for data_key in ['pop_map', 'landuse_maps', 'capacitance_maps', 'flow_map']:
            anim.append(np.load(str(dir_sub_path / f'{title}_{data_key}.npy')))

#  %% plot animation
phd_util.plt_setup(dark=False)
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
# prepare lists for data
anims = [anim_a, anim_b, anim_c]  #
gs = [graph_a, graph_b, graph_c]  #
ax_themes = ['Historic', 'Mesh-like', 'Tree-like']
xs = []
ys = []
ls = []
pops = []
flows = []
caps = []
lus = []
txts_iters = []
txts_flows = []
txts_lus = []
txts_caps = []

# append data for each graph
# frame the layouts for consistency across axes
x_extent = 1500
bottom_buffer = 50
y_extent = x_extent + bottom_buffer
for ax, anim, graph, ax_theme in zip(axes, anims, gs, ax_themes):
    # unpack
    pop_map, landuse_maps, capacitance_maps, flow_map = anim
    # extract xs and ys
    pos = {}
    for n, d in graph.nodes(data=True):
        pos[n] = (d['x'], d['y'])
    x_arr = np.array([v[0] for v in pos.values()])
    y_arr = np.array([v[1] for v in pos.values()])
    # setup ax
    ax.set_axis_off()
    # centre x
    x_min, x_max = (np.min(x_arr), np.max(x_arr))
    x_shift = -x_min + (x_extent - x_max) / 2
    x_arr = x_arr + x_shift
    # centre y
    y_min, y_max = (np.min(y_arr), np.max(y_arr))
    y_shift = -y_min + (y_extent - y_max) / 2 + bottom_buffer
    y_arr = y_arr + y_shift
    # append arrays
    xs.append(x_arr)
    ys.append(y_arr)
    # setup lines using similar logic
    lines = []
    for s, e in graph.edges():
        s_x = graph.nodes[s]['x'] + x_shift
        s_y = graph.nodes[s]['y'] + y_shift
        e_x = graph.nodes[e]['x'] + x_shift
        e_y = graph.nodes[e]['y'] + y_shift
        lines.append([[s_x, s_y], [e_x, e_y]])
    ls.append(lines)
    # set limits
    ax.set_xlim(0, x_extent)
    ax.set_ylim(0, y_extent)
    # generic empty placeholder arrays for nodes
    empty_size_arr = np.full(x_arr.shape[0], 0.0)
    # ax theme and iteration number
    txts_iters.append(ax.text(100,
                              y_extent - 100,
                              ax_theme,
                              fontsize=25))
    # aggregated flows text and symbol
    ax.scatter(x_extent - 100,
               y_extent - 85,
               s=[150],
               fc=(0, 0, 0, 0),
               ec=(20 / 255, 20 / 255, 20 / 255, 0.8),
               lw=0.25)
    txts_flows.append(ax.text(x_extent - 150,
                              y_extent - 100,
                              '',
                              fontsize=15,
                              ha='right'))
    # draw caps first (underneath) other layers
    inner_caps = []
    for l_s in layer_specs:
        fc = [c for c in l_s['colour']]
        fc[-1] = 0.1  # opacity
        inner_caps.append(ax.scatter(x_arr,
                                     y_arr,
                                     s=empty_size_arr,
                                     fc=fc,
                                     ec=(1, 1, 1, 1),
                                     lw=0.25))
    caps.append(inner_caps)
    # draw flows underneath landuse dots
    flows.append(ax.scatter(x_arr,
                            y_arr,
                            s=empty_size_arr,
                            fc=(0, 0, 0, 0),
                            ec=(20 / 255, 20 / 255, 20 / 255, 0.8),
                            lw=0.25))
    # upper line of text
    ax.text(100,
            100,
            'landuses',
            fontsize=15,
            va='center')
    # lower line of text
    ax.text(100,
            20,
            'capacitances',
            fontsize=15,
            va='center')
    inner_lus = []
    inner_txt_lus = []
    inner_txt_caps = []
    # draw dots and remaining text
    for l_s in layer_specs:
        # landuse dots per landuse type
        inner_lus.append(ax.scatter(x_arr,
                                    y_arr,
                                    s=empty_size_arr,
                                    fc=l_s['colour'],
                                    ec=(1, 1, 1, 1),
                                    lw=0.5))
        # legend symbol and agg text for landuses
        ax.scatter(l_s['to_x'],
                   100,
                   s=[100],
                   c=l_s['colour'],
                   alpha=1)
        inner_txt_lus.append(ax.text(l_s['to_x'] + 40,
                                     100,
                                     '',
                                     fontsize=15,
                                     va='center'))
        # legend symbol and agg text for capacitances
        ax.scatter(l_s['to_x'],
                   20,
                   s=[160],
                   fc=l_s['colour'],
                   ec=(1, 1, 1, 1),
                   alpha=0.25)
        inner_txt_caps.append(ax.text(l_s['to_x'] + 40,
                                      20,
                                      '',
                                      fontsize=15,
                                      va='center'))
    lus.append(inner_lus)
    txts_lus.append(inner_txt_lus)
    txts_caps.append(inner_txt_caps)
    # draw population dots on top
    pops.append(ax.scatter(x_arr,
                           y_arr,
                           s=empty_size_arr,
                           c='#2e2e2e',
                           label='dwellings'))


def init_func():
    artists = []
    for ax_idx, (ax, lines) in enumerate(zip(axes, ls)):
        ax.set_facecolor('#ffffff')
        ax.add_collection(LineCollection(lines, color='#2e2e2e', linewidth=0.5))
        artists.append(pops[ax_idx])
        artists.append(flows[ax_idx])
        artists.append(txts_iters[ax_idx])
        artists.append(txts_flows[ax_idx])
        for i in range(len(layer_specs)):
            artists.append(caps[ax_idx][i])
            artists.append(lus[ax_idx][i])
            artists.append(txts_lus[ax_idx][i])
            artists.append(txts_caps[ax_idx][i])
    # return iterable of artists to be redrawn - ignored if no blit
    return artists


def animate_func(i):
    '''
    "set_array" for colour
    "set_offsets" for x, y
    "set_sizes" for size
    '''
    artists = []
    for ax_idx, (
            ax, anim, fl, ca, lu, po, it_txt, fl_txt, txt_lu, txt_cap, ax_theme) in enumerate(
        zip(axes, anims, flows, caps, lus, pops, txts_iters, txts_flows, txts_lus,
            txts_caps, ax_themes)):

        pop_map, landuse_maps, capacitance_maps, flow_map = anim

        # population and flow map are shared across all layers
        p = pop_map[i]
        po.set_sizes(p * 0.5)
        artists.append(po)

        f = flow_map[i]
        fl.set_sizes(f * 0.8)
        artists.append(fl)

        # update iteration text
        it_txt.set_text(f'{ax_theme}: {i + 1}')
        artists.append(it_txt)

        # update flows text
        fl_txt.set_text(f"agg'd flows: {flow_map[i].sum():.2n}")
        artists.append(fl_txt)

        # landuse and capacitance maps are distinct per landuse spec
        for idx, l_s in enumerate(layer_specs):
            c = capacitance_maps[idx][i]
            ca[idx].set_sizes(c * 500)
            artists.append(ca[idx])

            l = landuse_maps[idx][i]
            lu[idx].set_sizes(l * 50)
            artists.append(lu[idx])

            txt_lu[idx].set_text(f'{int(landuse_maps[idx][i].sum())}')
            artists.append(txt_lu[idx])

            txt_cap[idx].set_text(f'{capacitance_maps[idx][i].sum():.2f}')
            artists.append(txt_cap[idx])

    # return iterable of all modified or created artists - ignored if no blit
    return artists


#  %%
fig.suptitle(title)
fp = f'../phd-admin/PhD/part_3/images/predicted/{title}.mp4'

# for "frames" argument, int is piped to a range operator
anim = animation.FuncAnimation(fig,
                               func=animate_func,
                               frames=plot_iters,
                               init_func=init_func,
                               # delay between frames in milliseconds - default 200
                               interval=200,
                               blit=True)
ff_writer = animation.FFMpegWriter(fps=10,
                                   codec='h264',
                                   bitrate=-1,  # auto
                                   metadata={
                                       'title': title,
                                       'artist': 'Gareth Simons',
                                       'copyright': 'Gareth Simons'
                                   })
anim.save(fp, writer=ff_writer, dpi=218)  # iMac 5k = 218
