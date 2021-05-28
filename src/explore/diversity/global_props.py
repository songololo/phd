# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import stats

from src import util_funcs


# db connection params
db_config = {
    'host': 'localhost',
    'port': 5433,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}
df = util_funcs.load_data_as_pd_df(
    db_config,
    ['pop_id',
     'city_name',
     'city_type',
     'city_area',
     'city_population',
     'city_species_count',
     'city_species_unique',
     'city_streets_len',
     'city_intersections_count'],
    'analysis.city_boundaries_150',
    'WHERE city_population IS NOT NULL ORDER BY pop_id')
Style = util_funcs.Style()

# %% plot population vs. total number of POI
# clear previous figures and set matplotlib defaults
util_funcs.plt_setup()
fig, axes = plt.subplots(1, 2, sharey='row', figsize=(6, 3))
# data
# d = df.dropna(how='all')
pop = df.city_population.values
species_count = df.city_species_count.values
count_dens = pop / species_count
# sizes
pop_log = np.log10(pop)
pop_log_norm = plt.Normalize()(pop_log)
count_dens_norm = plt.Normalize()(count_dens)
# curve fit to a powerlaw and plot
x_fit_line = np.arange(pop.min(), pop.max())


def powerFunc(x, c, z):
    return c * np.power(x, z)


# power fit - curve fit takes x_data and y_data
popt, _ = optimize.curve_fit(powerFunc, pop, species_count)
c = popt[0]
z = popt[1]
axes[0].plot(x_fit_line,
             powerFunc(x_fit_line, c, z),
             label=f'$POI=cP^z$ where $c={round(c, 2)}\ z={round(z, 2)}$',
             alpha=Style.alt_col_hl1_a,
             color=Style.alt_col_hl1,
             linestyle='-',
             linewidth=3)
# log log linear regression: linregress takes x, y
slope, intercept, rvalue, pvalue, stderr = stats.linregress(np.log(pop), np.log(species_count))
axes[0].plot(x_fit_line,
             np.exp(slope * np.log(x_fit_line) + intercept),
             label=f'$log(POI)=m log(P) + c$ where $m={round(slope, 2)}\ c={round(intercept, 2)}$',
             alpha=Style.alt_col_hl2_a,
             color=Style.alt_col_hl2,
             linestyle='-',
             linewidth=3)
# plot the area and population
scat = axes[0].scatter(x=pop,
                       y=species_count,
                       c=count_dens,
                       cmap=Style.alt_cmap,
                       s=pop_log_norm * 80 + 10,
                       marker='.',
                       edgecolors='white',
                       linewidths=0.2,
                       zorder=2)
axes[0].axhline(500,
                linewidth=0.4,
                color='dimgrey',
                linestyle='--')
axes[0].text(10 ** 7.5,
             530,
             'P.O.I. falloff at pop $x_{min}=5000$',
             horizontalalignment='right',
             fontdict={'size': 5})
axes[0].set_xscale('log')
axes[0].set_xlabel('Population per town / city')
axes[0].set_xlim(5000, 10 ** 7.5)
axes[0].set_yscale('log')
axes[0].set_ylabel('Points of Interest per town / city')
axes[0].set_ylim(100, 10 ** 6)
axes[0].legend(loc=2)
cbar = plt.colorbar(scat, aspect=50, pad=0.02)
cbar.set_label(label='Persons per Point Of Interest',
               loc='bottom')
# second ax
# plot the density against area
axes[1].scatter(x=count_dens,
                y=species_count,
                c=count_dens,
                cmap=Style.alt_cmap,
                s=pop_log_norm * 80 + 10,
                marker='.',
                edgecolors='white',
                linewidths=0.2,
                zorder=2)
axes[1].axhline(500,
                linewidth=0.4,
                color='dimgrey',
                linestyle='--')
# density colour fill
dens_col = plt.cm.viridis_r(count_dens_norm)
axes[1].vlines(x=count_dens,
               ymin=0,
               ymax=species_count,
               colors=dens_col,
               alpha=0.15,
               linewidths=pop_log_norm + .4,
               zorder=1)
axes[1].set_xlabel('Persons per Point of Interest')
axes[1].set_xlim(3, 60)
path = f'../phd-doc/doc/part_2/diversity/images/global_poi_count_pop.pdf'
plt.savefig(path)

# %% plot population vs. unique number of POI
# clear previous figures and set matplotlib defaults
util_funcs.plt_setup()
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
# data
# d = df.dropna(how='all')
pop = df.city_population.values
species_unique = df.city_species_unique.values
unique_dens = species_unique / pop
# sizes
pop_log = np.log(pop)
pop_log_norm = (pop_log - pop_log.min()) / (pop_log.max() - pop_log.min())
# curve fit to a powerlaw and plot
x_fit_line = np.arange(pop.min(), pop.max())


def powerFunc(x, c, z):
    return c * np.power(x, z)


# use log-log linear OLS as starting point
popt, _ = optimize.curve_fit(powerFunc, pop, species_unique)
c = popt[0]
z = popt[1]
axes[0].plot(x_fit_line,
             powerFunc(x_fit_line, c, z),
             label=f'$S=cA^z$ where $c={round(c, 2)}\ z={round(z, 2)}$',
             color=Style.def_col_hl1,
             alpha=Style.def_col_hl1_a,
             linestyle='-',
             linewidth=3)
# plot the area and population
axes[0].scatter(x=pop,
                y=species_unique,
                c=pop_log,
                cmap=Style.def_cmap,
                s=pop_log_norm * 80 + 10,
                marker='.',
                edgecolors='white',
                linewidths=0.2, zorder=2)
axes[0].axhline(620,
                linewidth=0.4,
                color='dimgrey',
                linestyle='--')
axes[0].text(5200,
             616,
             'taxonomic limit ($616$)',
             verticalalignment='bottom',
             fontdict={'size': 5})
axes[0].set_xscale('log')
axes[0].set_xlabel('Population per town / city')
axes[0].set_xlim(5000, 10 ** 7.5)
axes[0].set_yscale('log')
axes[0].set_ylabel('Unique Points of Interest per town / city')
axes[0].set_ylim(40, 1500)
axes[0].legend(loc=2)


# curve fit to semi-log model and plot
def semiLogFunc(x, c, z):
    return c + z * np.log(x)


popt, _ = optimize.curve_fit(semiLogFunc, pop, species_unique)
c = popt[0]
z = popt[1]
axes[1].plot(x_fit_line,
             semiLogFunc(x_fit_line, c, z),
             label=f'$S=c+z log(A)$ where $c={round(c, 2)}\ z={round(z, 2)}$',
             alpha=Style.def_col_hl2_a,
             color=Style.def_col_hl2,
             linestyle='-',
             linewidth=3)
# plot the area and population
scat = axes[1].scatter(x=pop,
                y=species_unique,
                c=pop_log,
                cmap=Style.def_cmap,
                s=pop_log_norm * 80 + 10,
                marker='.',
                edgecolors='white',
                linewidths=0.2,
                zorder=2)
axes[1].axhline(620,
                linewidth=0.4,
                color='dimgrey',
                linestyle='--')
axes[1].text(5200,
             620,
             'taxonomic limit ($616$)',
             verticalalignment='bottom',
             fontdict={'size': 5})
axes[1].set_xscale('log')
axes[1].set_xlabel('Population per town / city')
axes[1].set_xlim(5000, 10 ** 7.5)
# axes[0].set_yscale('log')
axes[1].set_ylabel('Unique Points of Interest per town / city')
axes[1].set_ylim(40, 840)
axes[1].legend(loc=2)
cbar = plt.colorbar(scat, aspect=50, pad=0.02)
cbar.set_label(label='$log$ population per town / city',
               loc='bottom')

path = f'../phd-doc/doc/part_2/diversity/images/global_poi_unique_pop.pdf'
plt.savefig(path)
