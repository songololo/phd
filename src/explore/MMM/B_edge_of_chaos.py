# %%

'''
EDGE OF CHAOS
- double tier layercake on MMM
'''

import os

os.environ['CITYSEER_QUIET_MODE'] = '1'
from _blocks import generate_graph
from layercake_c import mmm_nested_split, mmm_nested_doubles

fast_layer = {
    'cap_step': 0.5,
    'dist_threshold': 600,
    'pop_threshold': 200,
    'explore_rate': 0.9,
    'comp_rate': 0.9
}

slow_layer = {
    'cap_step': 0.2,
    'dist_threshold': 1200,
    'pop_threshold': 300,
    'explore_rate': 0.2,
    'comp_rate': 0.2
}

#  %% fast - slow/fast - fast composition
iters = 200
spans = 200
graph = generate_graph(spans)

hybrid_layers = (fast_layer.copy(), slow_layer.copy())

title = ''
theme = 'mmm_hybrid_fast_slow'
path = f'./src/explore/MMM/exploratory_plots/{theme}_split.png'

mmm_nested_split(graph,
                 iters,
                 hybrid_layers,
                 _title=title,
                 theme=theme,
                 path=path,
                 dark=True)

path = f'./src/explore/MMM/exploratory_plots/{theme}_doubles.png'
mmm_nested_doubles(graph,
                   iters,
                   hybrid_layers,
                   _title=title,
                   theme=theme,
                   path=path,
                   dark=True)
