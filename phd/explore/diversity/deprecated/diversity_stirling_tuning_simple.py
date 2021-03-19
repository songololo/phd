import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from scipy.stats import entropy

def stirling_diversity(uses_unique, uses_probs, alpha=1, beta=0.5):
    diversity = 0
    count_1 = count_2 = count_3 = 0
    for i in range(len(uses_unique) - 1):
        a = uses_unique[i]
        a_proportion = uses_probs[i]
        j = i + 1
        while j < len(uses_unique):
            b = uses_unique[j]
            b_proportion = uses_probs[j]
            if a[:1] != b[:1]:
                w = 1
                count_3 += 1
            elif a[1:2] != b[1:2]:
                w = 0.66
                count_2 += 1
            # since these are unique, no need to check remaining char
            else:
                w = 0.33
                count_1 += 1
            diversity += w**alpha * (a_proportion * b_proportion)**beta
            j += 1
    return diversity

def hill(uses_unique, uses_probs, exp):

    diversity = 0
    diversity_primary = 0
    diversity_secondary = 0
    diversity_tertiary = 0
    diversity_composite = 0
    for i in range(len(uses_unique) - 1):
        a = uses_unique[i]
        a_proportion = uses_probs[i]
        j = i + 1
        while j < len(uses_unique):
            b = uses_unique[j]
            b_proportion = uses_probs[j]
            if a[:1] != b[:1]:
                diversity_primary += i ** exp
                diversity += i ** exp
            elif a[1:2] != b[1:2]:
                diversity_secondary += i ** exp
                diversity += i ** exp
            # since these are unique, no need to check remaining char
            else:
                diversity_tertiary += i ** exp
                diversity += i ** exp
            j += 1
    return diversity

cases = [
    'aaa', 'aab', 'aac', 'aad',

    'aba', 'aca', 'ada', 'aea',

    'baa', 'caa', 'daa', 'eaa'
    ]

arrs = []
# level 1 - strictly unique
for i in range(1, 5):
    arrs.append(cases[:i])

# level 2 - strictly unique
for i in range(1, 5):
    arrs.append(cases[4:4+i])

# level 3 - strictly unique
for i in range(1, 5):
    arrs.append(cases[8:8+i])

# level 1 and 2 and 3 - strictly unique
for i in range(1, len(cases) + 1):
    arrs.append(cases[:i])

# level 3 n unique out of len homogeneous
for i in range(1, len(cases)+1):
    arr = cases[:i] + ['aaa'] * (len(cases) - i)
    arrs.append(arr)

vals = []
for count, arr in enumerate(arrs):
    arr = np.array(arr)
    arr_uniq, arr_counts = np.unique(arr, return_counts=True)
    arr_probs = arr_counts / len(arr)

    print('')
    print(arr)
    print('--------------')

    # 0 and 0 reduces to variety
    variety = stirling_diversity(arr_uniq, arr_probs, alpha=0, beta=0)
    vals.append([variety, 'variety: α=0, β=0', count])
    print('Variety:', round(variety, 3))

    # 1 and 0 reduces to disparity
    disparity = stirling_diversity(arr_uniq, arr_probs, alpha=1, beta=0)
    print('Disparity:', round(disparity, 3))
    vals.append([disparity, 'disparity: α=1, β=0', count])

    for i in range(1, 10, 1):
        i = i/10
        d = stirling_diversity(arr_uniq, arr_probs, alpha=1, beta=i)
        vals.append([d, f'α=1, β={i}', count])

    # 1 and 1 is base stirling diversity
    diversity = stirling_diversity(arr_uniq, arr_probs, alpha=1, beta=1)
    print('Diversity:', round(diversity, 3))
    vals.append([diversity, 'diversity: α=1, β=1', count])

    # 0 and 1 reduces to balance (half-gini)
    balance = stirling_diversity(arr_uniq, arr_probs, alpha=0, beta=1)
    print('Balance:', round(balance, 3))
    vals.append([balance, 'balance: α=0, β=1', count])

    # used to have additive, but additive doesn't work for homogeneous collectins
    # i.e. only performs as intended for fully unique sets

col = ['result', 'div_measure', 'strings']
df = pd.DataFrame(columns=col, data=vals)

ax = sns.factorplot(x='strings', y='result', hue='div_measure', data=df, legend_out=False, scale=0.4, size=10)
plt.xticks(rotation=90, visible=False)
plt.yscale('log')
plt.legend(loc='upper left')
plt.savefig('./diversity_stirling_tuning_simple.png', dpi=300)

