# %%
'''
ARTEFACTS AND AMNESIA
Outputs a series of plots showing how the model reacts to various parameter settings
The plots are split into bands showing how the model reacts to various pressure gradients,
e.g. increasing, decreasing, or gradients to left or right.
'''
import os

os.environ['CITYSEER_QUIET_MODE'] = '1'

from src.explore.MMM._blocks import generate_graph
from src.explore.MMM.layercake_d import mmm_single

# defaults
_cap_step = 0.2
_dist_threshold = 1200
_pop_threshold = 100
_explore_rate = 0.25
_comp_rate = 0.25
_echo_rate = 0.25

# capacitance step series
iters = 200
spans = 200
graph = generate_graph(spans)

titles = []
specs = []
for ax_n, cap_step in enumerate([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]):
    titles.append(f'MMM cap step: {cap_step}')
    specs.append({
        # lower cap steps are reticent to exploration
        'cap_step': cap_step,
        'dist_threshold': _dist_threshold,
        'pop_threshold': _pop_threshold,
        'comp_rate': _comp_rate,
        'explore_rate': _explore_rate,
        'echo_rate': _echo_rate
    })

theme = 'mmm_cap_step'
path = f'./explore/MMM/exploratory_plots/{theme}.png'
mmm_single(graph, iters, _layer_specs=specs, _title=titles, theme=theme, path=path, figsize=(40, 20), dark=True)

# population threshold series
iters = 200
spans = 200
graph = generate_graph(spans)

titles = []
specs = []
for ax_n, pop_thresh in enumerate([10, 25, 50, 100, 200, 500, 1000]):
    titles.append(f'MMM population thresh.: {pop_thresh}')
    specs.append({
        # lower cap steps are reticent to exploration
        'cap_step': _cap_step,
        'dist_threshold': _dist_threshold,
        'pop_threshold': pop_thresh,
        'comp_rate': _comp_rate,
        'explore_rate': _explore_rate,
        'echo_rate': _echo_rate
    })

theme = 'mmm_pop_thresh'
path = f'./explore/MMM/exploratory_plots/{theme}.png'
mmm_single(graph, iters, _layer_specs=specs, _title=titles, theme=theme, path=path, figsize=(40, 20), dark=True)

# distance threshold series
iters = 200
spans = 200
graph = generate_graph(spans)

titles = []
specs = []
for ax_n, dist_thresh in enumerate([100, 200, 400, 600, 800, 1200, 1600]):
    titles.append(f'MMM dist threshold: {dist_thresh}m')
    specs.append({
        'cap_step': _cap_step,
        'dist_threshold': dist_thresh,
        'pop_threshold': _pop_threshold,
        'comp_rate': _comp_rate,
        'explore_rate': _explore_rate,
        'echo_rate': _echo_rate
    })

theme = 'mmm_dist_threshold'
path = f'./explore/MMM/exploratory_plots/{theme}.png'
mmm_single(graph, iters, _layer_specs=specs, _title=titles, theme=theme, path=path, figsize=(40, 20), dark=True)

# comp_rate rate series
iters = 200
spans = 200
graph = generate_graph(spans)

titles = []
specs = []
for ax_n, comp_rate in enumerate([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]):
    titles.append(f'MMM comp_rate: {comp_rate}')
    specs.append({
        'cap_step': _cap_step,
        'dist_threshold': _dist_threshold,
        'pop_threshold': _pop_threshold,
        'comp_rate': comp_rate,
        'explore_rate': _explore_rate,
        'echo_rate': _echo_rate
    })

theme = 'mmm_comp_rate'
path = f'./explore/MMM/exploratory_plots/{theme}.png'
mmm_single(graph, iters, _layer_specs=specs, _title=titles, theme=theme, path=path, figsize=(40, 20), dark=True)

# exploration rate series
iters = 200
spans = 200
graph = generate_graph(spans)

titles = []
specs = []
for ax_n, explore_rate in enumerate([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]):
    titles.append(f'MMM exploration rate: {explore_rate}')
    specs.append({
        'cap_step': _cap_step,
        'dist_threshold': _dist_threshold,
        'pop_threshold': _pop_threshold,
        'comp_rate': _comp_rate,
        'explore_rate': explore_rate,
        'echo_rate': _echo_rate
    })

theme = 'mmm_exploration_rate'
path = f'./explore/MMM/exploratory_plots/{theme}.png'
mmm_single(graph, iters, _layer_specs=specs, _title=titles, theme=theme, path=path, figsize=(40, 20), dark=True)

# echos rate series
iters = 200
spans = 200
graph = generate_graph(spans)

titles = []
specs = []
for ax_n, echo_rate in enumerate([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]):
    titles.append(f'MMM echo rate: {echo_rate}')
    specs.append({
        'cap_step': _cap_step,
        'dist_threshold': _dist_threshold,
        'pop_threshold': _pop_threshold,
        'comp_rate': _comp_rate,
        'explore_rate': _explore_rate,
        'echo_rate': echo_rate
    })

theme = 'mmm_echo_rate'
path = f'./explore/MMM/exploratory_plots/{theme}.png'
mmm_single(graph, iters, _layer_specs=specs, _title=titles, theme=theme, path=path, figsize=(40, 20), dark=True)
