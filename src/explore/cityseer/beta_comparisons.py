# %% plot beta falloffs
import matplotlib.pyplot as plt
import numpy as np
from cityseer.metrics import networks

from src import util_funcs

#  %%
def avg_distances(beta):
    # looking for the average weight from 0 to d_max based on an impedance curve to d_max
    d = networks.distance_from_beta(beta)
    # area under the curve from 0 to d_max
    a = ((np.exp(-beta * d) - 1) / -beta)
    # divide by base (distance) for height, which gives weight
    w = a / d
    # then solve for the avg_d
    avg_d = -np.log(w) / beta
    # otherwise return the array
    return avg_d

# solve for beta using: log(1/y)/d where y = 0.01831563888873418
# or, per Elsa's comment: beta = 4 / d_tau
w_min = 0.01831563888873418
distances = np.array([100, 200, 400, 800, 1600])
betas_a = np.log(1 / w_min) / distances
betas_b = 4 / distances
assert np.allclose(betas_a, betas_b)
assert np.allclose(betas_a,
                   np.array([0.04, 0.02, 0.01, 0.005, 0.0025]),
                   atol=0.001,
                   rtol=0.0)
avg_ds = avg_distances(betas_a)

# %%
util_funcs.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
for d_max, beta, avg_d in zip(distances, betas_a, avg_ds):
    distances_arr = []
    for d in range(0, d_max + 1):
        distances_arr.append(d)
    distances_arr = np.array(distances_arr)
    y_falloff = np.exp(-beta * distances_arr)
    if d_max == 1600:
        spacer_a = '\ '
    else:
        spacer_a = '\ \ \ '
    if d_max < 800:
        spacer_b = '\ \ \ \ \ '
    elif d_max < 1600:
        spacer_b = '\ \ \ '
    else:
        spacer_b = '\ '
    ax.plot(distances_arr,
            y_falloff,
            lw=1.25,
            label=r'$d_{max}=' + f'{d_max}' + spacer_a +
                  f'\ \\beta={round(beta, 4)}' + spacer_b +
                  f'\ \\mu={round(avg_d)}m$'
            )
# add w_min
plt.axhline(y=w_min, ls='--', lw=0.5, c='grey')
ax.text(1500, 0.035, '$w_{min}$')

ax.set_xticks(distances)
ax.set_xticklabels(['100', '200', '$d_{max}=400$', '$d_{max}=800$', '$d_{max}=1600$'])
ax.set_xlim([0, 1600])
ax.set_xlabel('distance in metres $d$')
ax.set_ylim([0, 1.0])
ax.set_ylabel('effective weight $w$')
ax.set_title('$w = exp(-\\beta \\cdot d)\ \ \ \\beta=4/d_{max}$')
ax.legend(loc='upper right', labelspacing=0.8)
plt.savefig('../phd-doc/doc/images/cityseer/gravity_decay.pdf')
plt.savefig('../phd-doc/doc/images/centrality/gravity_decay.pdf')
plt.savefig('../phd-doc/doc/images/diversity/gravity_decay.pdf')

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
