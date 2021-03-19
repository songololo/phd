'''
Compares various mixed-use measures

'''

import numpy as np
import numba

#os.environ['NUMBA_WARNINGS'] = '1'
#os.environ['NUMBA_DEBUG_ARRAY_OPT_STATS'] = '1'


@numba.njit(parallel=True)
def deduce_unique_species(classes, distances, max_dist=1600):
    '''
    Sifts through the classes and returns unique classes, their counts, and the nearest distance to each respective type
    Only considers classes within the max distance
    Uses the closest specimen of any particular land-use as opposed to the average distance
    e.g. if a person where trying to connect to two nearest versions of type A and B
    '''
    # check that classes and distances are the same length
    if len(classes) != len(distances):
        raise ValueError('NOTE -> the classes array and the distances array need to be the same length')
    # not using np.unique because this doesn't take into account that certain classes exceed the max distance
    # prepare arrays for aggregation
    unique_count = 0
    classes_unique_raw = np.full(len(classes), np.nan)
    classes_counts_raw = np.full(len(classes), 0)  # int array
    classes_nearest_raw = np.full(len(classes), np.inf)  # coerce to float array
    # iterate all classes
    for i in range(len(classes)):
        d = distances[i]
        # check for valid entry - in case raw array is passed where unreachable verts have be skipped - i.e. np.inf
        if not np.isfinite(d):
            continue
        # first check that this instance doesn't exceed the maximum distance
        if d > max_dist:
            continue
        # if it doesn't, get the class
        c = classes[i]
        # iterate the unique classes
        # NB -> only parallelise the inner loop
        # if parallelising the outer loop it generates some very funky outputs.... beware
        for j in range(len(classes_unique_raw)):
            u_c = classes_unique_raw[j]
            # if already in the unique list, then increment the corresponding count
            if c == u_c:
                classes_counts_raw[j] += 1
                # check if the distance to this copy is closer than the prior least distance
                if d < classes_nearest_raw[j]:
                    classes_nearest_raw[j] = d
                break
            # if no match is encountered by the end of the list (i.e. np.nan), then add it:
            if np.isnan(u_c):
                classes_unique_raw[j] = c
                classes_counts_raw[j] += 1
                classes_nearest_raw[j] = d
                unique_count += 1
                break

    classes_unique = np.full(unique_count, np.nan)
    classes_counts = np.full(unique_count, 0)
    classes_nearest = np.full(unique_count, np.inf)
    for i in range(unique_count):
        classes_unique[i] = classes_unique_raw[i]
        classes_counts[i] = classes_counts_raw[i]
        classes_nearest[i] = classes_nearest_raw[i]

    return classes_unique, classes_counts, classes_nearest


@numba.njit
def dist_filter(cl_unique_arr, cl_counts_arr, cl_nearest_arr, max_dist):
    # first figure out how many valid items there are
    c = 0
    for i, d in enumerate(cl_nearest_arr):
        if d <= max_dist:
            c += 1
    # create trimmed arrays
    cl_unique_arr_trim = np.full(c, np.nan)
    cl_counts_arr_trim = np.full(c, 0)
    # then copy over valid data
    # don't parallelise - would cause issues
    c = 0
    for i, d in enumerate(cl_nearest_arr):
        if d <= max_dist:
            cl_unique_arr_trim[c] = cl_unique_arr[i]
            cl_counts_arr_trim[c] = cl_counts_arr[i]
            c += 1

    return cl_unique_arr_trim, cl_counts_arr_trim


@numba.njit
def beta_weights(arr, beta):
    return np.exp(beta * arr)


@numba.njit(parallel=True)
def gini_simpson(classes_counts):
    '''
    Gini-Simpson
    Gini transformed to 1 − λ
    Probability that two individuals picked at random do not represent the same species (Tuomisto)

    Ordinarily:
    D = 1 - sum(p**2) where p = Xi/N

    Bias corrected:
    D = 1 - sum(Xi/N * (Xi-1/N-1))
    '''
    N = classes_counts.sum()
    G = 0
    # catch potential division by zero situations
    if N < 2:
        return G
    # compute bias corrected gini-simpson
    for c in classes_counts:
        G += c / N * (c - 1) / (N - 1)
    return 1 - G


@numba.njit(parallel=True)
def shannon(classes_counts):
    '''
    Entropy
    p = Xi/N
    S = -sum(p * log(p))
    Uncertainty of the species identity of an individual picked at random (Tuomisto)
    '''
    N = classes_counts.sum()
    H = 0
    # catch potential division by zero situations
    if N == 0:
        return H
    # compute
    for a in classes_counts:
        p = a / N  # the probability of this class
        H += p * np.log(p)  # sum entropy
    return -H  # remember negative


@numba.njit(parallel=True)
def hill_div(classes_counts, q):
    '''
    Hill numbers - express actual diversity as opposed e.g. to Gini-Simpson (probability) and Shannon (information)

    exponent at 1 results in undefined because of 1/0 - but limit exists as exp(entropy)
    Ssee "Entropy and diversity" by Lou Jost

    Exponent at 0 = variety - i.e. count of unique species
    Exponent at 1 = unity
    Exponent at 2 = diversity form of simpson index
    '''

    N = classes_counts.sum()
    # catch potential division by zero situations
    if N == 0:
        return 0
    # hill number defined in the limit as the exponential of information entropy
    if q == 1:
        H = 0
        for a in classes_counts:
            p = a / N  # the probability of this class
            H += p * np.log(p)  # sum entropy
        return np.exp(-H)  # return exponent of entropy
    # otherwise use the usual form of Hill numbers
    else:
        D = 0
        for a in classes_counts:
            p = a / N  # the probability of this class
            D += p ** q  # sum
        return D ** (1 / (1 - q))  # return as equivalent species


@numba.njit(parallel=True)
def hill_div_phylo(classes_counts, class_dist, q, beta=-0.005):
    '''
    Based on unified framework for species diversity in Chao, Chiu, Jost 2014
    See table on page 308 and surrounding text

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the walking distance to a landuse impacts how strongly it contributes to diversity

    The weighting is based on the nearest of each landuse
    This is debatably most relevant to q=0
    '''

    # catch potential division by zero situations
    N = classes_counts.sum()
    if N == 0:
        return 0

    # find T
    T = 0
    for i in range(len(classes_counts)):
        weight = np.exp(class_dist[i] * beta)
        a = classes_counts[i] / N
        T += weight * a

    # hill number defined in the limit as the exponential of information entropy
    if q == 1:
        PD_lim = 0  # using the same variable name as non limit version causes errors for parallel
        # get branch lengths and class abundances
        for i in range(len(classes_counts)):
            weight = np.exp(class_dist[i] * beta)
            a = classes_counts[i] / N
            PD_lim += weight * a / T * np.log(a / T)  # sum entropy
        # return exponent of entropy
        PD_lim = np.exp(-PD_lim)
        return PD_lim #/ T
    # otherwise use the usual form of Hill numbers
    else:
        PD = 0
        # get branch lengths and class abundances
        for i in range(len(classes_counts)):
            weight = np.exp(class_dist[i] * beta)
            a = classes_counts[i] / N
            PD += weight * (a / T)**q  # sum
        # once summed, apply q
        PD = PD ** (1 / (1 - q))
        return PD #/ T


@numba.njit(parallel=True)
def hill_div_func(classes_counts, class_dist, q, beta=-0.005):
    '''
    Based on unified framework for species diversity in Chao, Chiu, Jost 2014
    See table on page 308 and surrounding text

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the walking distance to a landuse impacts how strongly it contributes to diversity

    Functional diversity takes the pairwise form, thus distances are based on pairwise i to j distances via the node k

    This is different to the non-pairwise form of the phylogenetic version which simply takes singular distance k to i
    '''

    # catch potential division by zero situations
    N = classes_counts.sum()
    if N < 2:
        return 0

    # calculate Q
    Q = 0
    for i in range(len(classes_counts)):
        weight_i = class_dist[i]
        a_i = classes_counts[i] / N
        for j in range(len(classes_counts)):
            # only need to examine the pair if j < i, otherwise double-counting
            if j >= i:
                break
            a_j = classes_counts[j] / N
            weight_j = class_dist[j]
            # pairwise distances
            d_ij = np.exp((weight_i + weight_j) * beta)
            Q += d_ij * a_i * a_j

    # if in the limit, use exponential
    if q == 1:
        FD_lim = 0  # using the same variable name as non limit version causes errors for parallel
        for i in range(len(classes_counts)):
            weight_i = class_dist[i]
            a_i = classes_counts[i] / N
            for j in range(len(classes_counts)):
                # only need to examine the pair if j < i, otherwise double-counting
                if j >= i:
                    break
                a_j = classes_counts[j] / N
                weight_j = class_dist[j]
                # pairwise distances
                d_ij = np.exp((weight_i + weight_j) * beta)
                FD_lim += d_ij * a_i * a_j / Q * np.log(a_i * a_j / Q)  # sum
        # once summed
        FD_lim = np.exp(-FD_lim)
        return FD_lim ** (1 / 2)  # (FD_lim / Q) ** (1 / 2)
    # otherwise conventional form
    else:
        FD = 0
        for i in range(len(classes_counts)):
            weight_i = class_dist[i]
            a_i = classes_counts[i] / N
            for j in range(len(classes_counts)):
                # only need to examine the pair if j < i, otherwise double-counting
                if j >= i:
                    break
                a_j = classes_counts[j] / N
                weight_j = class_dist[j]
                # pairwise distances
                d_ij = np.exp((weight_i + weight_j) * beta)
                FD += d_ij * (a_i * a_j / Q) ** q  # sum
        FD = FD ** (1 / (1 - q))
        return FD ** (1 / 2)  # (FD / Q) ** (1 / 2)


@numba.njit
def pairwise_disparity_weight(c_i, c_j, weight_1=1/3, weight_2=2/3, weight_3=3/3):
    '''
    Note that this is based on OS POI data, and the disparity methods work accordingly

    Weight can be topological, e.g. 1, 2, 3 or 1, 2, 5 or 0.25, 0.5, 1
    OR weighted per Warwick and Clarke based on decrease in taxonomic diversity
    Warwick and Clarke taxonomic distance - Clarke & Warwick: Weighting step lengths for taxonomic distinctness
    note that this doesn't necessarily add much benefit to topological distance
    also that it constrains the measure to the specific index

    in the case of OS - POI, there are 9 categories with 52 subcategories and 618 unique types
    therefore (see diversity_weights.py)
    in [618, 52, 9]
    out steps [0.916, 0.827, 0.889]
    additive steps [0.916, 1.743, 2.632]
    scaled [34.801, 66.223, 100.0]
    i.e. 1, 2, 3
    '''
    # calculate 3rd level disparity
    a_rem = c_i % 1000000
    b_rem = c_j % 1000000
    # if the classes diverge at the highest level, return the heaviest weight
    if c_i - a_rem != c_j - b_rem:
        return weight_3
    else:
        # else calculate 2nd level disparity, etc.
        a_small_rem = c_i % 10000
        b_small_rem = c_j % 10000
        if a_rem - a_small_rem != b_rem - b_small_rem:
            return weight_2
        # else default used - which means the disparity is at level 1
        else:
            return weight_1


@numba.njit(parallel=True)
def hill_div_disparity_os_poi(classes_unique, classes_counts, q):
    '''
    this is the same as above functional diversity, but uses species disparity weights
    '''

    # catch potential division by zero situations
    N = classes_counts.sum()
    if N < 2:
        return 0

    # find Q
    Q = 0
    for i in range(len(classes_counts)):
        a_i = classes_counts[i] / N
        for j in range(len(classes_counts)):
            # only need to examine the pair if j < i, otherwise double-counting
            if j >= i:
                break
            a_j = classes_counts[j] / N
            class_w = pairwise_disparity_weight(classes_unique[i], classes_unique[j])
            Q += class_w * a_i * a_j

    # if in the limit, use exponential
    if q == 1:
        FD_lim = 0  # using the same variable name as non limit version causes errors for parallel
        for i in range(len(classes_counts)):
            a_i = classes_counts[i] / N
            for j in range(len(classes_counts)):
                # only need to examine the pair if j < i, otherwise double-counting
                if j >= i:
                    break
                a_j = classes_counts[j] / N
                class_w = pairwise_disparity_weight(classes_unique[i], classes_unique[j])
                FD_lim += class_w * a_i * a_j / Q * np.log(a_i * a_j / Q)  # sum
        FD_lim = np.exp(-FD_lim)
        return FD_lim ** (1 / 2)  # (FD_lim / Q) ** (1 / 2)
    # otherwise conventional form
    else:
        FD = 0
        for i in range(len(classes_counts)):
            a_i = classes_counts[i] / N
            for j in range(len(classes_counts)):
                # only need to examine the pair if j < i, otherwise double-counting
                if j >= i:
                    break
                a_j = classes_counts[j] / N
                class_w = pairwise_disparity_weight(classes_unique[i], classes_unique[j])
                FD += class_w * (a_i * a_j / Q) ** q  # sum
        FD = FD ** (1 / (1 - q))
        return FD ** (1 / 2)  # (FD / Q) ** (1 / 2)


@numba.njit(parallel=True)
def raos_quad(classes_unique, classes_counts, alpha=1, beta=1):
    '''
    Rao's quadratic - bias corrected and based on disparity

    Sum of weighted pairwise products

    Note that Stirling's diversity is a rediscovery of Rao's quadratic diversity
    Though adds alpha and beta exponents to tweak weights of disparity dij and pi * pj, respectively
    This is a hybrid of the two, i.e. including alpha and beta options and adjusted for bias
    Rd = sum(dij * Xi/N * (Xj/N-1))

    Behaviour is controlled using alpha and beta exponents
    0 and 0 reduces to variety (effectively a count of unique types)
    0 and 1 reduces to balance (half-gini - pure balance, no weights)
    1 and 0 reduces to disparity (effectively a weighted count)
    1 and 1 is base stirling diversity / raos quadratic
    '''
    # catch potential division by zero situations
    N = classes_counts.sum()
    if N < 2:
        return 0

    R = 0  # variable for additive calculations of distance * p1 * p2
    for i in range(len(classes_unique)):
        # parallelise only inner loop
        for j in range(len(classes_counts)):
            # only need to examine the pair if j < i, otherwise double-counting
            if j >= i:
                break
            p_i = classes_counts[i] / N  # place here to catch division by zero for single element
            p_j = classes_counts[j] / (N - 1)  # bias adjusted
            # calculate 3rd level disparity
            w = pairwise_disparity_weight(classes_unique[i], classes_unique[j])
            R += w**alpha * (p_i * p_j)**beta
    return R


@numba.njit(parallel=True)
def quadratic_diversity_decomposed(classes_unique, classes_probs, num, weight_1=1, weight_2=2, weight_3=3):
    '''
    Based on a paper that untangles quadratic forms of diversity indices
    Not really used here but kept for posterity's sake...
    This version is not bias adjusted
    '''
    rao_stirling = 0
    disparity = 0
    balance = 0
    for i in range(len(classes_unique) - 1):
        a = classes_unique[i]
        a_proportion = classes_probs[i]
        j = i + 1
        while j < len(classes_unique):
            b = classes_unique[j]
            b_proportion = classes_probs[j]
            # calculate 3rd level disparity
            a_rem = a % 1000000
            b_rem = b % 1000000
            if a - a_rem != b - b_rem:
                w = weight_3
            else:
                # else calculate 2ndd level disparity
                a_small_rem = a % 10000
                b_small_rem = b % 10000
                if a_rem - a_small_rem != b_rem - b_small_rem:
                    w = weight_2
                # else default used - which means the disparity is at level 1
                else:
                    w = weight_1
            rao_stirling += w * (a_proportion * b_proportion)
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
    for i in range(len(classes_unique) - 1):
        a = classes_unique[i]
        a_proportion = classes_probs[i]
        j = i + 1
        while j < len(classes_unique):
            b = classes_unique[j]
            b_proportion = classes_probs[j]
            # calculate 3rd level disparity
            a_rem = a % 1000000
            b_rem = b % 1000000
            if a - a_rem != b - b_rem:
                w = weight_3
            else:
                # else calculate 2ndd level disparity
                a_small_rem = a % 10000
                b_small_rem = b % 10000
                if a_rem - a_small_rem != b_rem - b_small_rem:
                    w = weight_2
                # else default used - which means the disparity is at level 1
                else:
                    w = weight_1
            balance_factor += (w - species_distinctness) * (a_proportion * b_proportion - d)
            j += 1
    balance_factor = 2 * balance_factor
    combined = simpson_index * species_distinctness + balance_factor
    return simpson_index, species_distinctness, balance_factor, combined
