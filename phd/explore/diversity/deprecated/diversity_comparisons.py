import numpy as np
from scipy.stats import entropy
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def simpson(uses_probs):
    '''
    Simpson is sum of squares of probabilities
    Gini is inverse of Simpson
    '''
    diversity = 0
    for i in uses_probs:
        diversity += i**2
    return diversity


def hill(uses_probs, exp):
    '''
    Hill numbers - express actual diversity as opposed e.g. to entropy
    Exponent at 0 = variety - i.e. count of unique species
    Exponent at 1 = unity
    Exponent at 2 = diversity form of simpson index
    '''
    # exponent at 1 results in undefined because of 1/0 - but limit exists as exp(entropy)
    # see "Entropy and diversity" by Lou Jost
    if exp == 1:
        return math.exp(entropy(uses_probs))
    else:
        diversity = 0
        for i in uses_probs:
            diversity += i**exp
        return diversity ** (1 / (1 - exp))


def balance(uses_probs):
    '''
    Balance is the sum of the product of pairwise probabilities, i.e. sum(prob_A * prob_B) for all pairs
    Balance is equivalent to half gini i.e. (1 - simpson) / 2
    '''
    diversity = 0
    for i in range(len(uses_probs) - 1):
        a_proportion = uses_probs[i]
        j = i + 1
        while j < len(uses_probs):
            b_proportion = uses_probs[j]
            diversity += a_proportion * b_proportion
            j += 1
    return diversity


def disparity(uses_unique, variety=False):
    '''
    Disparity is the sum of pairwise weighted distance disparities between all pairs (no probabilities involved)
    Variety is effectively a non-weighted version, which is equivalent to (N^2-N) / 2.
    Pass True to variety flag or see next function
    '''
    diversity = 0
    for i in range(len(uses_unique) - 1):
        a = uses_unique[i]
        j = i + 1
        while j < len(uses_unique):
            # variety is unweighted
            if variety:
                diversity += 1
            # disparity is weighted
            else:
                b = uses_unique[j]
                if a[:1] != b[:1]:
                    w = 1
                elif a[1:2] != b[1:2]:
                    w = 0.66
                # since these are unique, no need to check remaining char
                else:
                    w = 0.33
                diversity += w
            j += 1
    return diversity


def variety(uses_unique):
    '''
    Variety is effectively a non-weighted version of disparity (above), which is equivalent to (N^2-N) / 2
    '''
    return ((len(uses_unique)**2 - len(uses_unique)) / 2)


def stirling_diversity(uses_unique, uses_probs, alpha=1, beta=1):
    """
    Sum of weighted pairwise products

    Behaviour is controlled using alpha and beta exponents
    0 and 0 reduces to variety
    0 and 1 reduces to balance (half-gini)
    1 and 0 reduces to disparity
    1 and 1 is base stirling diversity

    An alpha of 1 makes sense (it is effectively just weighting the weights)

    Beta has to be tuned to somewhere between behaving as variety and behaving balance.
    When in full balance mode, it starts behaving counter-intuitively for larger,
    more complex assemblages because of the increasingly smaller proportions
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
            diversity += w**alpha * (a_proportion * b_proportion)**beta
            j += 1
    return diversity


def taxonomic_distinctness(uses_unique, uses_probs):
    '''
    Clarke and Warwick
    '''
    diversity = 0
    agg = 0
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
            diversity += w * (a_proportion * b_proportion)
            agg += a_proportion * b_proportion
            j += 1
    if agg:
        return diversity / agg
    return 0

def taxonomic_diversity(uses_unique, uses_probs, num):
    '''
    Clarke and Warwick
    '''
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
            diversity += w * (a_proportion * b_proportion)
            j += 1
    if num - 1 != 0:
        return diversity / ((num * (num - 1)) / 2)
    return 0

cases = [
    'aaa', 'aab', 'aac', 'aad', 'aae', 'aaf', 'aag', 'aah', 'aai', 'aaj', 'aak', 'aal', 'aam', 'aan', 'aao', 'aap', 'aaq', 'aar', 'aas', 'aat', 'aau', 'aav', 'aaw', 'aax', 'aay',

    'aba', 'aca', 'ada', 'aea', 'afa', 'aga', 'aha', 'aia', 'aja', 'aka', 'ala', 'ama', 'ana', 'aoa', 'apa', 'aqa', 'ara', 'asa', 'ata', 'aua', 'ava', 'awa', 'axa', 'aya', 'aza',

    'baa', 'caa', 'daa', 'eaa', 'faa', 'gaa', 'haa', 'iaa', 'jaa', 'kaa', 'laa', 'maa', 'naa', 'oaa', 'paa', 'qaa', 'raa', 'saa', 'taa', 'uaa', 'vaa', 'waa', 'xaa', 'yaa', 'zaa'
    ]

arrs = []
# level 1 - strictly unique
for i in range(1, 26):
    arrs.append(cases[:i])

# level 2 - strictly unique
for i in range(1, 26):
    arrs.append(cases[25:25+i])

# level 3 - strictly unique
for i in range(1, 26):
    arrs.append(cases[50:50+i])

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
    num = len(arr)
    arr_uniq, arr_counts = np.unique(arr, return_counts=True)
    arr_probs = arr_counts / len(arr)

    print('')
    print(arr)
    print('--------------')

    s = entropy(arr_probs)
    vals.append([s, 'shannon entropy', count])
    print('Shannon entropy:', round(s, 3))

    g = simpson(arr_probs)  # inverse simpson
    vals.append([1 - g, 'gini / simpson index', count])
    print('Gini Simpson index:', round(g, 3))

    h = hill(arr_probs, 0)
    vals.append([h, 'hill @exp 0', count])
    print('Hill exponent 0:', round(h, 3))

    h = hill(arr_probs, 1)
    vals.append([h, 'hill @exp 1: undefined: limit exp(entropy)', count])
    print('Hill exponent 1: exponent(shannon entropy): ', round(h, 3))

    h = hill(arr_probs, 2)
    vals.append([h, 'hill @exp 2', count])
    print('Hill exponent 2:', round(h, 3))

    stirling = stirling_diversity(arr_uniq, arr_probs)
    vals.append([stirling, 'stirling diversity: α=1, β=1', count])
    print('Stirling Diversity:', round(stirling, 3))

    stirling_tuned = stirling_diversity(arr_uniq, arr_probs, beta=0.5)
    vals.append([stirling_tuned, 'tuned stirling diversity: α=1, β=0.5', count])
    print('Stirling Diversity @ β = 0.5:', round(stirling_tuned, 3))

    taxonomic_dist = taxonomic_distinctness(arr_uniq, arr_probs)
    vals.append([taxonomic_dist, 'taxonomic distinctness', count])
    print('taxonomic Distinctness:', round(taxonomic_dist, 3))

    taxonomic_div = taxonomic_diversity(arr_uniq, arr_probs, num)
    vals.append([taxonomic_div, 'taxonomic diversity', count])
    print('Taxonomic Diversity:', round(taxonomic_div, 3))

col = ['result', 'div_measure', 'strings']
df = pd.DataFrame(columns=col, data=vals)

ax = sns.factorplot(x='strings', y='result', hue='div_measure', data=df, legend_out=False, scale=0.4, size=10)
plt.xticks(rotation=90, visible=False)
plt.yscale('log')
plt.legend(loc='upper left')
plt.savefig('./diversity_comparisons.png', dpi=300)

