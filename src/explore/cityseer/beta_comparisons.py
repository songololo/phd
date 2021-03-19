# %% plot beta falloffs
import matplotlib.pyplot as plt
import numpy as np
from src import phd_util

# %%
# solve for beta using: log(1/y)/d where y = 0.01831563888873418
w_min = 0.01831563888873418
betas = []
for d_max in [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]:
    betas.append(np.log(1 / w_min) / d_max)
# [0.08, 0.04, 0.02666666666666667, 0.02, 0.013333333333333334, 0.01, 0.006666666666666667, 0.005, 0.0033333333333333335, 0.0025]


# %%
phd_util.plt_setup()

fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
w_min = 0.01831563888873418
# set the betas
betas = []
distances = [100, 200, 400, 800, 1600]
for d_max in distances:
    beta = np.log(w_min) / d_max
    distances_arr = []
    for d in range(0, d_max + 1):
        distances_arr.append(d)
    distances_arr = np.array(distances_arr)
    y_falloff = np.exp(beta * distances_arr)
    ax.plot(distances_arr, y_falloff, label=f'$\\beta={round(beta, 4)}$')

# add w_min
plt.axhline(y=w_min, ls='--', lw=0.5, c='grey')
ax.text(1500, 0.035, '$w_{min}$')

ax.set_xticks(distances)
ax.set_xticklabels(['100', '200', '$d_{max}=400$', '$d_{max}=800$', '$d_{max}=1600$'])
ax.set_xlim([0, 1600])
ax.set_xlabel('Distance in Metres')
ax.set_ylim([0, 1.0])
ax.set_ylabel('Weighting')
ax.legend(loc='upper right', title='$exp(\\beta \\cdot d[i,j])$')

plt.savefig('../phd-admin/PhD_latex/1/images/cityseer/gravity_decay.png', dpi=300)
plt.savefig('./explore/2_cityseer/plots/betas.png', dpi=300)

plt.show()

# %%

fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))

betas_arr = [0.0025, 0.005, 0.01, 0.02]
max_distance = 1601

distances_arr = []
for distance in range(0, max_distance):
    distances_arr.append(distance)
distances_arr = np.array(distances_arr)

harmonic_closeness = 1 / distances_arr
ax.plot(distances_arr, harmonic_closeness, label='harmonic closeness')

for d in [200, 400, 800, 1600]:
    linear_closeness = (d - distances_arr) / d
    ax.plot(distances_arr, linear_closeness, label=f'linear closeness ${d}' + '_{m}$')

# log_closeness = 1 - np.log(distances_arr)/np.log(400)
# ax.plot(distances_arr, log_closeness, label='log closeness')

# exp_closeness = 400 ** -(distances_arr / 400)
# ax.plot(distances_arr, exp_closeness, label='exponential closeness')

# accessibility = 1 / (distances_arr ** 0.1)
# ax.plot(distances_arr, accessibility, label='crude accessibility closeness')

# for i in range(1, 10, 1):
#    improved_closeness = i / distances_arr
#    ax.plot(distances_arr, improved_closeness, label=f'improved {i} nodes')

# add the cutoff lines
ax.vlines(x=[200, 400, 800, 1600], ymin=0, ymax=0.05, linewidth=0.8, linestyles='dashed',
          colors=['red', 'green', 'orange', 'blue'])

ax.set_xlim([0, 1601])
ax.set_xlabel('Distance in Metres')
ax.set_ylim([0, 1])
ax.set_ylabel('Weighting')
ax.legend(loc='upper right', title='$exp(-\\beta \\cdot d[i,j])$')
plt.show()
