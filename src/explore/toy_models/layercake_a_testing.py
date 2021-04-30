# %%
import os

os.environ['CITYSEER_QUIET_MODE'] = '1'

from src.explore.toy_models import generate_graph
from src.explore.toy_models import mmm_single

iters = 200
graph = generate_graph(200)

layer_specs = {
    'cap_step': 0.2,
    'dist_threshold': 1200,
    'pop_threshold': 100,
    'explore_rate': 0.5,
    'spill_rate': 0
}

theme = 'test_layercake'
path = f'./src/explore/H_mmm/exploratory_plots/{theme}.png'
mmm_single(graph,
           iters,
           _layer_specs=layer_specs,
           theme=theme,
           path=path,
           dark=True)
