# %%
import os

os.environ['CITYSEER_QUIET_MODE'] = '1'

from src.explore.MMM._blocks import generate_graph
from src.explore.MMM.layercake_d import mmm_single

iters = 250
graph = generate_graph(_spans=200)

layer_specs = {
    'cap_step': 0.5,
    'dist_threshold': 1200,
    'pop_threshold': 200,
    'explore_rate': 0.75,
    'comp_rate': 0.5,
    'echo_rate': 0.75
}

theme = 'test_layercake'
path = f'./explore/H_mmm/exploratory_plots/{theme}.png'
mmm_single(graph,
           iters,
           _layer_specs=layer_specs,
           theme=theme,
           path=path,
           dark=True)