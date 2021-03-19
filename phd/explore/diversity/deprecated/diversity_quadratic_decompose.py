import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def quadratic_diversity_decomposed(uses_unique, uses_probs, num, flat=False):
    """
    Explores paper "On the measurement of species diversity incorporating species differences
    by
    Kenichiro Shimatani

    Decomposes quadratic diversity (full matrix version of Stirling Diversity, no alpha or beta) into:
    Simpson index (D) x Species Distinctness (Q+) + Balance Factor (B)
    In these example cases the Species Distinctness is maximal (no redundant features)
    """
    stirling = 0
    disparity = 0
    balance = 0
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
            if flat:
                w = 1
            stirling += w * (a_proportion * b_proportion)
            disparity += w
            balance += a_proportion * b_proportion
            j += 1

    simpson_index = 2 * balance
    n = 0
    if num > 1:
        n = 1/(num*(num-1)/2)
    species_distinctness = n * disparity
    d = n * balance
    balance_factor = 0
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
            if flat:
                w = 1
            balance_factor += (w - species_distinctness) * (a_proportion * b_proportion - d)
            j += 1
    balance_factor = 2 * balance_factor
    combined = simpson_index * species_distinctness + balance_factor
    return stirling, simpson_index, species_distinctness, balance_factor, combined

cases = [
    'aaa', 'aab', 'aac', 'aad', 'aae', 'aaf', 'aag', 'aah', 'aai', 'aaj', 'aak', 'aal', 'aam', 'aan', 'aao', 'aap', 'aaq', 'aar', 'aas', 'aat', 'aau', 'aav', 'aaw', 'aax', 'aay',

    'aba', 'aca', 'ada', 'aea', 'afa', 'aga', 'aha', 'aia', 'aja', 'aka', 'ala', 'ama', 'ana', 'aoa', 'apa', 'aqa', 'ara', 'asa', 'ata', 'aua', 'ava', 'awa', 'axa', 'aya', 'aza',

    'baa', 'caa', 'daa', 'eaa', 'faa', 'gaa', 'haa', 'iaa', 'jaa', 'kaa', 'laa', 'maa', 'naa', 'oaa', 'paa', 'qaa', 'raa', 'saa', 'taa', 'uaa', 'vaa', 'waa', 'xaa', 'yaa', 'zaa'
    ]

arrs = []
# level 1 - strictly unique
for i in range(1, 26):
    arrs.append((cases[:i], False))

# level 2 - strictly unique
for i in range(0, 26):
    arrs.append((cases[25:25+i], False))

# level 3 - strictly unique
for i in range(0, 26):
    arrs.append((cases[50:50+i], False))

# level 1 and 2 and 3 - strictly unique
for i in range(0, len(cases)):
    arrs.append((cases[0:i], False))

# level 3 n unique out of len homogeneous
for i in range(0, len(cases)):
    arr = cases[0:i] + ['aaa'] * (len(cases) - i)
    arrs.append((arr, True))

vals = []
for count, case in enumerate(arrs):
    arr, flat = case
    num = len(arr)
    arr = np.array(arr)
    arr_uniq, arr_counts = np.unique(arr, return_counts=True)
    arr_probs = arr_counts / len(arr)

    print('')
    print(arr)
    print('--------------')

    # 0 and 0 reduces to variety
    stirling, simpson_index, species_distinctness, balance_factor, combined = \
        quadratic_diversity_decomposed(arr_uniq, arr_probs, num, flat)

    print('Simpson Diversity:', round(simpson_index, 3))
    vals.append([simpson_index, 'Simpson diversity', count])

    print('Stirling Diversity:', round(stirling, 3))
    vals.append([stirling, 'Stirling diversity - half Qaudratic', count])

    print('Species Distinctness:', round(species_distinctness, 3))
    vals.append([species_distinctness, 'species distinctness', count])

    print('Balance Factor:', round(balance_factor, 3))
    # round balance_factor so as not to plot micro numbers otherwise chart scale distorted
    vals.append([round(balance_factor, 5), 'balance factor', count])

    print('Combined:', round(combined, 3))
    vals.append([combined, 'combined = D x Q+ + B', count])

col = ['result', 'div_measure', 'strings']
df = pd.DataFrame(columns=col, data=vals)

ax = sns.factorplot(x='strings', y='result', hue='div_measure', data=df, legend_out=False, scale=0.4, size=10)
plt.xticks(rotation=90, visible=False)
#plt.yscale('log')
plt.legend(loc='upper left')
plt.savefig('./diversity_quadratic_decompose.png', dpi=300)

