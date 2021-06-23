# %%
'''
Example locations for plotting vis-a-vis cluttering / maps
'''
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Proj, transform
from sklearn.preprocessing import StandardScaler

from src import util_funcs
from src.explore import plot_funcs
from src.explore.signatures import sig_models
from src.explore.theme_setup import data_path, weights_path
from src.explore.theme_setup import generate_theme

#  %%
locations = [
    'Carnaby Street',
    'Seven Dials',
    'Primrose Hill',
    'Crouch End',
    'Ikea Edmonton',
    'Croydon'
]

indices = [
    '1_C8BB3E22-3BB9-4C5C-A463-C73B36B83846',  # carnaby street
    '1_042BA54D-8726-4DAD-BD05-45475148D18F',  # seven dials
    '1_B25102D8-1353-458F-A96D-BAC48A2407E5',  # primrose hill
    '1_E6AB1D3C-80D3-4324-917F-1581EE0CA55E_3_1_CB1F8F8D-B8DD-4E23-AD4B-285FD388D95F',  # Crouch End
    '1_576FDF2D-61F1-4F0C-A44D-93DDCB5896B1',  # ikea
    '1_9A4426B0-12E6-42C4-B733-1137762F0440_3_1_D9A5397C-5432-4001-802E-83C556309431'  # Croydon
]
#  %%
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
table = df_20

X_raw, distances, labels = generate_theme(df_20, 'all_towns', bandwise=True, max_dist=800)
X_trans = StandardScaler().fit_transform(X_raw)

bng = Proj('epsg:27700')
lng_lat = Proj('epsg:4326')

#  %%
location_keys = {}
for loc, idx in zip(locations, indices):
    x, y = table.loc[idx, 'x'], table.loc[idx, 'y']
    lat, lng = transform(bng, lng_lat, x, y)
    location_keys[loc] = {
        'idx': idx,
        'iloc': table.index.get_loc(idx),
        'x': round(x, 1),
        'y': round(y, 1),
        'lng': round(lng, 3),
        'lat': round(lat, 3)
    }
    print(location_keys[loc])

#  %%
'''
Prepare split model
NB - this version uses bandwise=False
'''
# setup paramaters
epochs = 5
batch = 256
theme_base = f'VAE_e{epochs}'
n_d = len(distances)
split_input_dims = (int(2 * n_d), int(9 * n_d), int(2 * n_d))
split_latent_dims = (4, 6, 2)
split_hidden_layer_dims = ([24, 24, 24],
                           [32, 32, 32],
                           [8, 8, 8])
latent_dim = 6
seed = 0
beta = 1
cap = 8
lr = 1e-3

# setup model
vae = sig_models.SplitVAE(raw_dim=X_trans.shape[1],
                          latent_dim=latent_dim,
                          beta=beta,
                          capacity=cap,
                          epochs=epochs,
                          split_input_dims=split_input_dims,
                          split_latent_dims=split_latent_dims,
                          split_hidden_layer_dims=split_hidden_layer_dims,
                          theme_base=theme_base,
                          seed=seed,
                          name='VAE')
dir_path = pathlib.Path(
    weights_path / f'signatures_weights/seed_{seed}/{vae.theme}_epochs_{epochs}_batch_{batch}_train')
vae.load_weights(str(dir_path / 'weights'))
Z_mu, Z_log_var, Z = vae.encode(X_trans, training=False)

# %%
'''
Plots decoded landuses from latents for respective example locations
'''
# plots decoded exemplar landuses
util_funcs.plt_setup()
fig, axes = plt.subplots(1, len(indices), figsize=(7, 3.5))
names = [n for n in location_keys.keys()]
indices = [i['iloc'] for i in location_keys.values()]
z_keys = []
arrs = []
maxes = np.full(len(labels), 0)

for idx in range(len(indices)):
    loc_idx = indices[idx]
    z_state = Z_mu[loc_idx]
    z_key = np.array([[*z_state]])  # expects 2d
    z_keys.append(z_key)
    arr = vae.decode(z_key)
    # reshape
    arr = np.reshape(arr, (len(labels), len(distances)))
    maxes = np.maximum(maxes, np.max(np.abs(arr), axis=1))
    arrs.append(arr)

for ax_idx, ax in enumerate(axes):
    name = names[ax_idx]
    arr = arrs[ax_idx]
    norm_arr = np.copy(arr)
    for row_idx in range(arr.shape[0]):
        norm_arr[row_idx] = arr[row_idx] / maxes[row_idx]
    plot_funcs.plot_heatmap(ax,
                            heatmap=norm_arr,
                            row_labels=labels,
                            col_labels=distances,
                            set_row_labels=ax_idx == 0)
    ax.set_title(f'{name}')
    # prepare lng, lat
    lng = location_keys[name]["lng"]
    if lng < 0:
        lng = f'{lng:.3f}°W'
    else:
        lng = f'{lng:.3f}°E'
    lat = f'{location_keys[name]["lat"]:.3f}°N'
    # prepare codes
    loc_idx = indices[ax_idx]
    z_state = Z_mu[loc_idx]
    z_txt = [str(f'{z:.1f}') for z in z_state]
    z_txt = '|'.join(z_txt)
    ax.set_xlabel(f'Decoded from latents:\n{z_txt}\nFor location:\n{lng} {lat}', fontsize='xx-small', loc='left')
plt.suptitle('Landuse decodings for example locations')
path = f'../phd-doc/doc/images/signatures/vae_example_decodings.pdf'
plt.savefig(path, dpi=300)
