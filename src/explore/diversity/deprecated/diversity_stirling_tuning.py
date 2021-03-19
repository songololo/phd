import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def stirling_diversity(uses_unique, uses_probs, alpha=1, beta=0.5):
    """
    Sum of weighted pairwise products

    Behaviour is controlled using alpha and beta exponents
    0 and 0 reduces to variety (effectively a count of unique types)
    0 and 1 reduces to balance (half-gini - pure balance, no weights)
    1 and 0 reduces to disparity (effectively a weighted count)
    1 and 1 is base stirling diversity

    An alpha of 1 makes sense (it is effectively just weighting the weights)

    Beta has to be tuned to somewhere between behaving as variety and behaving balance.
    When in full balance mode, it starts behaving counter-intuitively for larger,
    more complex assemblages because of the increasingly smaller proportions
    see for example, that 1 plus 2 plus 3 level differences have a smaller end result
    than only level 3 differences (refer to plotted graph)

    Weight can be topological, e.g. 1, 2, 3 or 1, 2, 5 or 0.25, 0.5, 1
    OR weighted per Warwick and Clarke based on decrease in taxonomic diversity
    in the case of OS - POI, there are 9 categories with 52 subcategories and 618 unique types
    therefore:
    in [618, 52, 9]
    out steps [0.916, 0.827, 0.889]
    additive steps [0.916, 1.743, 2.632]
    scaled [34.801, 66.223, 100.0]
    i.e. 1, 2, 3
    """
    diversity = 0
    for i in range(len(uses_unique) - 1):
        a = uses_unique[i]
        a_proportion = uses_probs[i]
        j = i + 1
        while j < len(uses_unique):
            b = uses_unique[j]
            b_proportion = uses_probs[j]
            if a[:1] != b[:1]:
                w = 1
            elif a[1:2] != b[1:2]:
                w = 0.66
            # since these are unique, no need to check remaining char
            else:
                w = 0.33
            diversity += w ** alpha * (a_proportion * b_proportion) ** beta
            j += 1
    return diversity


cases = [
    'aaa', 'aab', 'aac', 'aad', 'aae', 'aaf', 'aag', 'aah', 'aai', 'aaj', 'aak', 'aal', 'aam', 'aan', 'aao', 'aap',
    'aaq', 'aar', 'aas', 'aat', 'aau', 'aav', 'aaw', 'aax', 'aay',

    'aba', 'aca', 'ada', 'aea', 'afa', 'aga', 'aha', 'aia', 'aja', 'aka', 'ala', 'ama', 'ana', 'aoa', 'apa', 'aqa',
    'ara', 'asa', 'ata', 'aua', 'ava', 'awa', 'axa', 'aya', 'aza',

    'baa', 'caa', 'daa', 'eaa', 'faa', 'gaa', 'haa', 'iaa', 'jaa', 'kaa', 'laa', 'maa', 'naa', 'oaa', 'paa', 'qaa',
    'raa', 'saa', 'taa', 'uaa', 'vaa', 'waa', 'xaa', 'yaa', 'zaa'
]

arrs = []
# level 1 - strictly unique
for i in range(1, 26):
    arrs.append(cases[:i])

# level 2 - strictly unique
for i in range(1, 26):
    arrs.append(cases[25:25 + i])

# level 3 - strictly unique
for i in range(1, 26):
    arrs.append(cases[50:50 + i])

# level 1 and 2 and 3 - strictly unique
for i in range(1, len(cases)):
    arrs.append(cases[0:i])

# level 3 n unique out of len homogeneous
for i in range(1, len(cases)):
    arr = cases[0:i] + ['aaa'] * (len(cases) - i)
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
    print('Variety:', round(variety, 3))
    vals.append([variety, 'variety: α=0, β=0', count])

    # 1 and 0 reduces to disparity
    disparity = stirling_diversity(arr_uniq, arr_probs, alpha=1, beta=0)
    print('Disparity:', round(disparity, 3))
    vals.append([disparity, 'disparity: α=1, β=0', count])

    for i in range(1, 10, 1):
        i = i / 10
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
plt.savefig('./diversity_stirling_tuning.png', dpi=300)
