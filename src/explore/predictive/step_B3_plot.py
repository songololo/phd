# %%
'''
STEP 3: plot animations of MMM in action across three scenarios

Uses predictive eating establishments model as a fitness function for CA simulation

Dwelling density
SELECT SUM(cents.dwelling) FROM
    (SELECT geom FROM analysis.city_boundaries_150 WHERE pop_id = 1) as bounds,
    (SELECT dwelling, geom FROM census_2011.census_centroids) as cents
    WHERE ST_Contains(bounds.geom, cents.geom);
-- 3707571

SELECT COUNT(nodes.id) FROM
    (SELECT id FROM analysis.nodes_20 WHERE city_pop_id = 1 AND within = true) as nodes;
-- 1097510
'''
import os

os.environ['CITYSEER_QUIET_MODE'] = '1'

import pathlib
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src import util_funcs
from src.explore.theme_setup import data_path, weights_path, generate_theme
from src.explore.toy_models import step_C1_graphs
from src.explore.predictive.step_B2_MMML_DNN import mmml_phd
from src.explore.predictive import pred_models

#  %%
'''
Three different landuses - compete with like-for-like vs. synergy with other landuses
'''
# '#9a0007'
# '#0064b7'
# '#0091ea'
# '#d32f2f'
layer_specs = [
    # commercial
    {
        'colour': [0 / 255, 155 / 255, 176 / 255, 1],
        'to_x': 450
    },
    # manufacturing
    {
        'colour': [184 / 255, 42 / 255, 158 / 255, 1],
        'to_x': 700
    },
    # food retail
    {
        'colour': [252 / 255, 74 / 255, 43 / 255, 1],
        'to_x': 950
    },
]

#  %% run models
'''
Three different graphs with distinctly different network topologies
'''
iters = 500
seed = 123
cap_step = 0.1
cap_jitter = 0.5
max_cap = 2.0
new_threshold = 0.5
dwell_density = 15  # assumes higher density - about 3.3 average for london

# towns vs london
location = 'london'

theme = f'MMML_{location}_hmmz'
title = f'{theme}_iter{iters}_seed{seed}_cs{cap_step}_cj{cap_jitter}_mc{max_cap}_nt{new_threshold}_dd{dwell_density}'

graph_a = step_C1_graphs.york_burb()
graph_b = step_C1_graphs.grid_ville()
graph_c = step_C1_graphs.suburb()

# prepare data for transformer
# required by ML models for mapping data space to prediction data space
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
X_raw, distances, labels = generate_theme(df_20, 'pred_sim', bandwise=False)

# load models into a dictionary
epochs = 100
models = {}
transformers = {}
target_distances = [400, 1600, 400]
target_col_templates = ['ac_commercial_{dist}', 'ac_manufacturing_{dist}', 'ac_retail_food_{dist}']
for col_template, target_dist in zip(target_col_templates, target_distances):
    # prepare model
    target_w_dist = col_template.format(dist=target_dist)
    reg = pred_models.LandUsePredictor(theme_base=f'{target_w_dist}_e{epochs}_{location}_{target_dist}')
    ml_model_path = pathlib.Path(weights_path / f'{reg.theme}')
    reg.load_weights(str(ml_model_path / 'weights'))
    reg.compile(optimizer='adam')
    # prepare transformer
    drop_cols = []
    for dist in distances:
        drop_cols.append(col_template.format(dist=dist))
    print(f'Dropping ancillary columns: {drop_cols}')
    X_dropped = X_raw.drop(columns=drop_cols)
    transformer = StandardScaler()
    transformer.fit(X_dropped)
    # set
    models[col_template] = {
        'target_dist': target_dist,
        'model': reg,
        'transformer': transformer
    }

column_order = ['c_node_harmonic_angular_{dist}',
                'c_node_betweenness_beta_{dist}',
                'ac_commercial_{dist}',
                'ac_manufacturing_{dist}',
                'ac_retail_food_{dist}',
                'cens_dwellings_{dist}']

# create, else load data
dir_path = pathlib.Path('./temp_data/')
if not dir_path.exists():
    dir_path.mkdir(exist_ok=True, parents=True)
if not pathlib.Path(dir_path / 'anim_ml_a' / f'{title}_dwell_map.npy').exists():
    anim_a = mmml_phd(graph_a,
                      iters,
                      distances,
                      dwell_density,
                      models,
                      column_order,
                      cap_step,
                      cap_jitter,
                      max_cap,
                      new_threshold,
                      seed)
    anim_b = mmml_phd(graph_b,
                      iters,
                      distances,
                      dwell_density,
                      models,
                      column_order,
                      cap_step,
                      cap_jitter,
                      max_cap,
                      new_threshold,
                      seed)
    anim_c = mmml_phd(graph_c,
                      iters,
                      distances,
                      dwell_density,
                      models,
                      column_order,
                      cap_step,
                      cap_jitter,
                      max_cap,
                      new_threshold,
                      seed)
    for anim, anim_key in zip([anim_a, anim_b, anim_c], ['a', 'b', 'c']):
        for data, data_key in zip(anim,
                                  ['dwell_map', 'landuse_maps', 'capacitance_maps']):
            dir_sub_path = pathlib.Path(dir_path / f'anim_ml_{anim_key}')
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
        dir_sub_path = pathlib.Path(dir_path / f'anim_ml_{anim_key}')
        for data_key in ['dwell_map', 'landuse_maps', 'capacitance_maps']:
            anim.append(np.load(str(dir_sub_path / f'{title}_{data_key}.npy')))

#  %% plot animation
util_funcs.plt_setup(dark=False)
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
# prepare lists for data
anims = [anim_a, anim_b, anim_c]  #
gs = [graph_a, graph_b, graph_c]  #
ax_themes = ['Historic', 'Mesh-like', 'Tree-like']
xs = []
ys = []
ls = []
dwells = []
caps = []
lus = []
txts_iters = []
txts_lus = []
txts_caps = []

# append data for each graph
# frame the layouts for consistency across axes
x_extent = 1500
bottom_buffer = 50
y_extent = x_extent + bottom_buffer
for ax, anim, graph, ax_theme in zip(axes, anims, gs, ax_themes):
    # unpack
    dwellings_map, landuse_maps, capacitance_maps = anim
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
    # draw dwelling dots on top
    dwells.append(ax.scatter(x_arr,
                             y_arr,
                             s=empty_size_arr,
                             c='#2e2e2e',
                             label='dwellings'))


def init_func():
    artists = []
    for ax_idx, (ax, lines) in enumerate(zip(axes, ls)):
        ax.set_facecolor('#ffffff')
        ax.add_collection(LineCollection(lines, color='#2e2e2e', linewidth=0.5))
        artists.append(dwells[ax_idx])
        artists.append(txts_iters[ax_idx])
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
            ax, anim, ca, lu, po, it_txt, txt_lu, txt_cap, ax_theme) in enumerate(
        zip(axes, anims, caps, lus, dwells, txts_iters, txts_lus,
            txts_caps, ax_themes)):

        dwellings_map, landuse_maps, capacitance_maps = anim

        # dwellings are shared across all layers
        p = dwellings_map[i]
        po.set_sizes(p * 0.1)
        artists.append(po)

        # update iteration text
        it_txt.set_text(f'{ax_theme}: {i + 1}')
        artists.append(it_txt)

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

            txt_cap[idx].set_text(f'{int(capacitance_maps[idx][i].sum())}')
            artists.append(txt_cap[idx])

    # return iterable of all modified or created artists - ignored if no blit
    return artists


#  %%
fig.suptitle(title)
fp = f'../phd-doc/doc/part_3/images/predicted/{title}.mp4'

# for "frames" argument, int is piped to a range operator
anim = animation.FuncAnimation(fig,
                               func=animate_func,
                               frames=iters,  # iters
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
