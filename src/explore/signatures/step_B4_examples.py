# %%
'''
Example locations for plotting vis-a-vis cluttering / maps
'''
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import phd_util
from src.explore import plot_funcs
from src.explore.signatures import sig_models, sig_model_runners
from src.explore.theme_setup import data_path, logs_path
from src.explore.theme_setup import generate_theme
from pyproj import Proj, transform
from sklearn.preprocessing import StandardScaler

#  %%
locations = [
    'Seven Dials',
    'Oxford Circus',
    'Primrose Hill',
    'Crouch End',
    'Hackney',
    'Angel',
    'Croydon',
    'Dagenham'
]

indices = [
    '1_042BA54D-8726-4DAD-BD05-45475148D18F',
    '1_FCA12886-4447-4C7A-BEDA-10E22E671BB0',
    '1_595E24E0-0F46-4634-929F-C854D6A9E7DB',
    '1_E6AB1D3C-80D3-4324-917F-1581EE0CA55E_3_1_CB1F8F8D-B8DD-4E23-AD4B-285FD388D95F',
    '1_B8F41B70-37B6-425C-9FDF-F31D6124D8F1_1_1_2648AB14-E08A-4BB7-807A-EADC7783FF9D',
    '1_60664D0F-DA94-4461-9076-DB8F305512D6',
    '1_9A4426B0-12E6-42C4-B733-1137762F0440',
    '1_406B2485-AA29-4331-8320-9CD3577B76F1_1_1_2906401A-8A53-4DAE-A70E-E3DF92EA7DF5'
]

df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
table = df_20
X_raw, distances, labels = generate_theme(table, 'all', bandwise=False)
trans_model = StandardScaler()
X_trans = trans_model.fit_transform(X_raw)
test_idx = phd_util.train_test_idxs(df_20, 200)  # 200 gives about 25%

bng = Proj('epsg:27700')
lng_lat = Proj('epsg:4326')

location_keys = {}
for loc, idx in zip(locations, indices):
    x, y = table.loc[idx, 'x'], table.loc[idx, 'y']
    lat, lng = transform(bng, lng_lat, x, y)
    location_keys[loc] = {
        'idx': idx,
        'iloc': table.index.get_loc(idx),
        'x': round(x, 1),
        'y': round(y, 1),
        'lng': round(lng, 4),
        'lat': round(lat, 4)
    }

#  %%
'''
Prepare split model
NB - this version uses bandwise=False
'''
# setup paramaters
epochs = 100
batch = 256
theme_base = f'VAE_examples_no_bandwise'
n_d = len(distances)
split_input_dims = (int(5 * n_d), int(18 * n_d), int(4 * n_d))
split_latent_dims = (6, 8, 2)
split_hidden_layer_dims = ([128, 128, 128],
                           [256, 256, 256],
                           [64, 64, 64])
latent_dim = 8
seed = 0
beta = 4
cap = 12
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
dir_path = pathlib.Path(f'./temp_weights/signatures_weights/seed_{seed}/{vae.theme}_epochs_{epochs}_batch_{batch}')
if not dir_path.exists():
    trainer = sig_model_runners.VAE_trainer(model=vae,
                                            X_samples=X_trans,
                                            labels=labels,
                                            distances=distances,
                                            logs_path=logs_path,
                                            epochs=epochs,
                                            batch=batch,
                                            val_split=0.2,
                                            best_loss=True,
                                            lr=lr,
                                            save_path=dir_path,
                                            test_indices=test_idx)
    trainer.train()
else:
    vae.load_weights(str(dir_path / 'weights'))
Z_mu, Z_log_var, Z = vae.encode(X_trans, training=False)

# %%
'''
Plots decoded landuses from latents for respective example locations
'''
# plots decoded exemplar landuses
phd_util.plt_setup()
fig, axes = plt.subplots(1, 8, figsize=(13, 6))
names = [n for n in location_keys.keys()]
indices = [i['iloc'] for i in location_keys.values()]
z_keys = []
arrs = []
maxes = np.full(len(labels), 0)

for idx in range(8):
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
                            norm_arr,
                            labels,
                            distances,
                            set_row_labels=ax_idx == 0)
    lng = location_keys[name]["lng"]
    if lng < 0:
        lng = f'{lng}°W'
    else:
        lng = f'{lng}°E'
    lat = f'{location_keys[name]["lat"]}°N'
    ax.set_title(f'{name}\n${lng}\ {lat}$', fontsize='small')
    # prepare codes
    loc_idx = indices[ax_idx]
    z_state = Z_mu[loc_idx]
    z_txt = [str(f'${z:.2f}$') for z in z_state]
    z_txt_a = str.join(", ", z_txt[:4])
    z_txt_b = str.join(", ", z_txt[4:])
    ax.set_xlabel(f'landuses decoded from latents:\n{z_txt_a} \n{z_txt_b}', fontsize='x-small', )
plt.suptitle('Landuse decodings for example locations')
path = f'../../phd-admin/PhD/part_3/images/signatures/vae_example_decodings.png'
plt.savefig(path, dpi=300)
