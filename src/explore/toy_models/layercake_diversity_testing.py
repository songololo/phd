# %%
import os

import matplotlib.pyplot as plt

from src import util_funcs

os.environ['CITYSEER_QUIET_MODE'] = '1'
from src.explore.toy_models import generate_graph
from src.explore.toy_models import mmm_wrap

iters = 50
spans = 200
graph = generate_graph(spans)

util_funcs.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(12, 20))

layer_spec = {
    'cap_step': 0.25,
    'dist_threshold': 800,
    'pop_threshold': 100,
    'spill_rate': 0.25,
    'explore_rate': 0.25,
    'innovation_rate': 0.25
}
mmm_wrap(ax, graph, layer_spec, iters, spans)

theme = 'test'
fig.suptitle(theme)
plt.savefig(f'./src/explore/H_mmm/exploratory_plots/{theme}.png', dpi=300)
plt.show()
