# %%
'''
Plots scatter and powerlaw info for global relationships / powerlaw etc.
Based on GLOBAL population and area
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import stats

from src import util_funcs

Style = util_funcs.Style()

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
     'city_area_petite',
     'city_population',
     'city_species_count',
     'city_species_unique',
     'city_streets_len',
     'city_intersections_count'],
    'analysis.city_boundaries_150',
    'WHERE city_population is not null ORDER BY pop_id ASC')

#  %% area vs population and density

util_funcs.plt_setup()

fig, axes = plt.subplots(1, 2, sharey='row', figsize=(7, 3.5))

# data
d = df.dropna(how='all')
pop = d.city_population.values
area = d.city_area_petite.values
area = area * 0.0001  # convert to hectare
dens = pop / area

# sizes
pop_log = np.log10(pop)
pop_log_norm = plt.Normalize()(pop_log)
dens_norm = plt.Normalize()(dens)

# curve fit to a powerlaw and plot
x_fit_line = np.arange(pop.min(), pop.max())


def powerFunc(x, c, a):
    return c * np.power(x, a)


fp1, fc1 = optimize.curve_fit(powerFunc, pop, area)  # use log-log linear OLS as starting point
axes[0].plot(x_fit_line, powerFunc(x_fit_line, *fp1),
             label=f'non-linear LS curve-fit: $\\alpha={round(fp1[1], 2)}$', alpha=Style.def_col_hl1_a,
             color=Style.def_col_hl1, linestyle='-', linewidth=3)
# linear regression on the log log
slope, intercept, rvalue, pvalue, stderr = stats.linregress(np.log(pop), np.log(area))
axes[0].plot(x_fit_line, np.exp(slope * np.log(x_fit_line) + intercept),
             label=f'log-log OLS: $\\alpha={round(slope, 2)}$', alpha=Style.def_col_hl2_a, color=Style.def_col_hl2,
             linestyle='-', linewidth=3)
# plot the area and population
axes[0].scatter(x=pop, y=area, c=dens, cmap=Style.def_cmap, s=pop_log_norm * 80 + 10,
                marker='.', edgecolors='white', linewidths=0.2, zorder=2)
# insert a line showing area threshold
axes[0].axhline(400, linewidth=0.4, color='dimgrey', linestyle='--')
axes[0].text(10 ** 7.5, 430, 'area falloff at pop $x_{min}=5000$', horizontalalignment='right',
             fontdict={'size': 5})
axes[0].set_xscale('log')
axes[0].set_xlabel('City Population')
axes[0].set_xlim(5000, 10 ** 7.5)
axes[0].set_yscale('log')
axes[0].set_ylabel('City Area (hectare - excluding parks)')
axes[0].set_ylim(100, 10 ** 5.5)
axes[0].legend(loc=2)

# plot the density against area
axes[1].scatter(x=dens,
                y=area,
                c=dens,
                cmap=Style.def_cmap,
                s=pop_log_norm * 80 + 10,
                marker='.',
                edgecolors='white',
                linewidths=0.2,
                zorder=2,
                rasterized=True)
axes[1].axhline(400, linewidth=0.4, color='dimgrey', linestyle='--')
# density colour fill
dens_col = plt.cm.plasma_r(dens_norm)
axes[1].vlines(x=dens, ymin=0, ymax=area, colors=dens_col, alpha=0.15,
               linewidths=pop_log_norm + .4, zorder=1)
# insert a line showing area threshold
axes[1].set_xlabel('City Density (persons per hectare)')
axes[1].set_xlim(10, 70)

path = f'../phd-doc/doc/part_3/predictive/images/area_pop_density.pdf'
plt.savefig(path, dpi=300)

plt.show()

#  %% plot powerlaw distributions for city population

# clear previous figures and set matplotlib defaults
util_funcs.plt_setup()

fig, axes = plt.subplots(1, 2, sharey='row', figsize=(7, 3.5))

# data
d = df.dropna(how='all')
pop = d.city_population.values

# plot the powerlaw checks
import powerlaw

# (-3.6101996064653328, 0.16366603647613442) for xmin = 11066
# (-5.536245962618196, 0.0858027317226928) for xmin = 10000
fit = powerlaw.Fit(pop, discrete=True)
pl_xmin = round(fit.power_law.xmin, 0)
pl_alpha = round(fit.power_law.alpha, 2)
powerlaw_fit = r'powerlaw: $x_{min}' + f'={pl_xmin}, \\alpha=-{pl_alpha}$'
ln_mu = round(fit.lognormal.mu, 2)
ln_sigma = round(fit.lognormal.sigma, 2)
lognormal_fit = f'lognormal: $\\mu={ln_mu}, \\sigma={ln_sigma}$'
# distribution compare returns two numbers:
# 1) R: Loglikelihood ratio of the two distributions’ fit to the data.
# If greater than 0, the first distribution is preferred.
# If less than 0, the second distribution is preferred.
# 2) p: Significance of R
print('distribution compare: power law vs. lognormal:', fit.distribution_compare('power_law', 'lognormal'))
# (-3.6101996064653328, 0.16366603647613442)

'''
https://www.quora.com/What-is-the-difference-between-a-probability-density-function-and-a-cumulative-distribution-function
A PDF answers the question: "How common are samples at exactly this value?"
A CDF answers the question "How common are samples that are less than or equal to this value?"
The CDF is the integral of the PDF.
'''
fit.power_law.plot_ccdf(ax=axes[0], label=powerlaw_fit, alpha=1, color=Style.alt_col_hl1, linestyle='--', linewidth=1)
fit.lognormal.plot_ccdf(ax=axes[0], label=lognormal_fit, alpha=1, color=Style.alt_col_hl2, linestyle='--', linewidth=1)
fit.plot_ccdf(ax=axes[0], original_data=False, label=f'data', alpha=0.5, color='green', linestyle='-', linewidth=3)
axes[0].set_xscale('log')
axes[0].set_xlabel(r'Population $x_{min}' + f'={pl_xmin}$')
axes[0].set_xlim(10 ** 4, 10 ** 7)
axes[0].set_yscale('log')
axes[0].set_ylabel(r'CCDF  $\overline{F}(x)=P(X>x)=1-F(x)$')
axes[0].set_ylim(10 ** -4, 2)
axes[0].legend(loc=3)

fit = powerlaw.Fit(pop, discrete=True, xmin=50000)
pl_xmin = round(fit.power_law.xmin, 0)
pl_alpha = round(fit.power_law.alpha, 2)
powerlaw_fit = r'powerlaw:  $x_{min}' + f'={pl_xmin}, \\alpha=-{pl_alpha}$'
ln_mu = round(fit.lognormal.mu, 2)
ln_sigma = round(fit.lognormal.sigma, 2)
lognormal_fit = f'lognormal: $\\mu={ln_mu}, \\sigma={ln_sigma}$'
print('distribution compare: power law vs. lognormal:', fit.distribution_compare('power_law', 'lognormal'))
'''
returns:
Loglikelihood ratio of the two distributions’ fit to the data. If greater than 0, the first distribution is preferred. If less than 0, the second distribution is preferred.
p - Significance of R
distribution compare: power law vs. lognormal: (-3.066214123591209, 0.19468594720893306)
distribution compare: power law vs. lognormal: (-0.2752671386000767, 0.7022809977778219)
'''
fit.power_law.plot_ccdf(ax=axes[1], label=powerlaw_fit, alpha=1, color=Style.alt_col_hl1, linestyle='--', linewidth=1)
fit.lognormal.plot_ccdf(ax=axes[1], label=lognormal_fit, alpha=1, color=Style.alt_col_hl2, linestyle='--', linewidth=1)
fit.plot_ccdf(ax=axes[1], original_data=False, label=f'data', alpha=0.5, color='green', linestyle='-', linewidth=3)
axes[1].set_xscale('log')
axes[1].set_xlabel(r'Population $x_{min}' + f'={pl_xmin}$')
axes[1].set_xlim(10 ** 4, 10 ** 7)
axes[1].legend(loc=3)

path = f'../phd-doc/doc/part_3/predictive/images/pop_scale_fit.pdf'
plt.savefig(path, dpi=300)

plt.show()
