# %%
import os

os.environ['CITYSEER_QUIET_MODE'] = '1'
print(os.curdir)

from src.explore.MMM._blocks import generate_graph
from src.explore.MMM.layercake_b import mmm_single

iters = 200
graph = generate_graph(200)

layer_specs = {
    'cap_step': 0.5,
    'dist_threshold': 1200,
    'pop_threshold': 400,
    'explore_rate': 0.5
}

theme = 'test_layercake'
path = f'./src/explore/H_mmm/exploratory_plots/{theme}.png'
mmm_single(graph,
           iters,
           _layer_specs=layer_specs,
           seed=False,
           theme=theme,
           path=path,
           dark=True)
