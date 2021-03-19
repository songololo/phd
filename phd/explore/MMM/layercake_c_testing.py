#%%
import os
os.environ['CITYSEER_QUIET_MODE'] = '1'

from explore.MMM._blocks import generate_graph
from explore.MMM.layercake_c import mmm_single

iters = 250
graph = generate_graph(_spans=200)

layer_specs = {
    'cap_step': 0.5,
    'dist_threshold': 1200,
    'pop_threshold': 100,
    'explore_rate': 0.5,
    'comp_rate': 0.5
}

theme = 'test_layercake'
path = f'./explore/H_mmm/exploratory_plots/{theme}.png'
mmm_single(graph,
           iters,
           _layer_specs=layer_specs,
           theme=theme,
           path=path,
           dark=True)
