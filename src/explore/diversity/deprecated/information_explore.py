"""
Aggregates measures per node based on network distances

"""
import logging

import numpy as np
from graph_tool.all import *
from numba import jit
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f'graph-tool version: {graph_tool.__version__}')


@jit(nopython=True, nogil=True)
def stirling_div_mod(uses_unique, uses_probs):
    mixedUses = 0  # variable for additive calculations of distance * p1 * p2
    count = 0
    w_1 = 0
    w_2 = 0
    w_3 = 0
    for i in range(len(uses_unique) - 1):
        a = uses_unique[i]
        a_proportion = uses_probs[i]
        j = i + 1
        while j < len(uses_unique):
            b = uses_unique[j]
            b_proportion = uses_probs[j]
            a_rem = a % 1000000
            b_rem = b % 1000000
            if a - a_rem != b - b_rem:
                w_3 += 1
                w = 1
            else:
                a_small_rem = a % 10000
                b_small_rem = b % 10000
                if a_rem - a_small_rem != b_rem - b_small_rem:
                    w_2 += 1
                    w = 0.66
                else:
                    w = 0.33
                    w_1 += 1
                # checking the last four digits not necessary here because complete matches are already computed
            # Stirling's distance weighted diversity index.
            # matching items have distance of 0 so don't compute for complete matches
            count += 1
            mixedUses += w * a_proportion
            mixedUses += w * b_proportion
            j += 1
    print('count', count)
    print('W1:', w_1, 'W2:', w_2, 'W3:', w_3, w_1 + w_2 + w_3)
    return mixedUses


@jit(nopython=True, nogil=True)
def stirling_div_uniq(uses_unique, uses_probs):
    mixedUses = 0  # variable for additive calculations of distance * p1 * p2
    count = 0
    w_1 = 0
    w_2 = 0
    w_3 = 0
    for i in range(len(uses_unique) - 1):
        a = uses_unique[i]
        a_proportion = uses_probs[i]
        j = i + 1
        while j < len(uses_unique):
            b = uses_unique[j]
            b_proportion = uses_probs[j]
            a_rem = a % 1000000
            b_rem = b % 1000000
            if a - a_rem != b - b_rem:
                w_3 += 1
                w = 1
            else:
                a_small_rem = a % 10000
                b_small_rem = b % 10000
                if a_rem - a_small_rem != b_rem - b_small_rem:
                    w_2 += 1
                    w = 0.66
                else:
                    w = 0.33
                    w_1 += 1
                # checking the last four digits not necessary here because complete matches are already computed
            # Stirling's distance weighted diversity index.
            # matching items have distance of 0 so don't compute for complete matches
            count += 1
            mixedUses += w * a_proportion * b_proportion
            j += 1
    print('count', count)
    print('W1:', w_1, 'W2:', w_2, 'W3:', w_3, w_1 + w_2 + w_3)
    return mixedUses


boring_arr = [2110191, 6340440, 10540736, 2030056, 2090141, 2090811, 9470699, 6340453, 10590732, 6340454, 3170245,
              3170245, 6340454, 6340455, 4240290, 6330416, 3170245, 7410542, 2080115, 10550746, 10590732, 4240305,
              6340459, 4240305, 3180255, 2080117, 2100156, 2600099, 10590732, 10590732, 9480677, 2600099, 2100156,
              10590732, 9480675, 10550746, 4240290, 10590732, 6340461, 6340459, 2150223]

exciting_arr = [9470663, 1020013, 9470661, 6340802, 2090151, 2090141, 2090150, 9480686, 2090151, 5270325, 9460659,
                9470661, 9470661, 9460656, 10540736, 3200268, 2100156, 5280368, 7400524, 9470663, 9470699, 3170813,
                2090811, 2100156, 2050078, 2050078, 4240293, 1020013, 5260322, 2100156, 9470663, 2100180, 9460659,
                4220279, 2100776, 9470699, 2100156, 2100156, 2100156, 2100156, 4220279, 9470665, 2100156, 1020034,
                9460660, 2090141, 3200269, 1020043, 9480716, 7370471, 2100182, 6340461, 9460656, 9480824, 2090147,
                9480690, 1020034, 1020034, 1020034, 3170245, 2110194, 2100156, 5290358, 4240293, 2100156, 1020013,
                6340459, 2100156, 1020020, 1020034, 1020034, 6340457, 6340802, 7420568, 9470662, 2100156, 5280330,
                9470699, 4220279, 9470705, 9480712, 2090141, 1020034, 2100180, 1020034, 2070100, 1020013, 2090154,
                1020034, 2110190, 7420599, 7420599, 2100156, 2090135, 2050078, 2100180, 1020018, 2090149, 2110192,
                9460656, 10590732, 9470661, 5280330, 9470672, 2090138, 2090141, 2090141, 2090141, 9470699, 2110190,
                9480716, 5280344, 4220279, 7370478, 9480675, 7410531, 2080117, 1020043, 1020018, 9470699, 1020018,
                2090138, 9470661, 2090138, 6340802, 2130793, 2130793, 2090141, 2090141, 2090141, 2090141, 9460656,
                9470661, 9480726, 9480677, 6340802, 9480725, 9470662, 2090138, 9480714, 2100156, 1020020, 5280330,
                5280344, 2100156, 7370473, 1020034, 2050078, 2100170, 9460656, 9480674, 2050078, 7370476, 5280344,
                9480714, 9470665, 2100180, 9480694, 2150230, 2150230, 7400524, 9480714, 9480716, 2100156, 7370473,
                2050078, 2100156, 2100156, 2100776, 2050078, 2100156, 2100156, 2100156, 2090154, 1020034, 9470662,
                2100156, 2130793, 1020034, 6340461, 9480726, 9480714, 1020013, 9460659, 9470662, 2100170, 2130793,
                7370785, 7370476, 5280344, 7370478, 4220277, 2100156, 9480686, 2100156, 9480763, 9480682, 7370481,
                9480714, 9480686, 9460656, 9480690, 2130212, 9490695, 2090147, 2090135, 2100156, 2100156, 10590732,
                2130212, 5260316, 2130212, 7410542, 4250315, 4250314, 4250315, 6350448, 6350452, 10590732, 1020013,
                4250312, 1020034, 10590732, 2090141, 2090141, 2090141, 2090141, 2090138, 2100156, 5280344, 2100156,
                2100180, 9480689, 2090147, 2100156, 10590732, 9470699, 2100156, 2090141, 4250312, 1020018, 2100156,
                1020018, 2150230, 2150230, 9480828, 2080117, 2100156, 2100156, 1020013, 1020013, 6340433, 9480680,
                1020018, 1020043, 1020018, 1020019, 2150230, 1010009, 6340802, 9480701, 1020034, 1020034, 1020034,
                2150230, 1020034, 5320395, 1020034, 1020034, 5280344, 6340802, 9460659, 9470672, 2100156, 9480717,
                9460656, 9460659, 9480726, 9460656, 6340802, 9480717, 9480726, 9480677, 9460659, 2100177, 10590732,
                9480726, 9480763, 1020018, 1020018, 4250312, 2100156, 4250312, 1020034, 9460659, 9460656, 2090150,
                2110190, 9460659, 1020034, 2110190, 2100156, 3170245, 5280345, 1020034, 1020034, 2090138, 2110190,
                9460656, 9480674, 9460659, 9460656, 9480701, 9480701, 9460656, 2090141, 2090141, 2090141, 2090141,
                2090141, 2090138]

test_boring_5 = [2110191, 6340440, 10540736, 2030056, 2090141]

test_boring_10 = [2110191, 6340440, 10540736, 2030056, 2090141, 2090811, 9470699, 6340453, 10590732, 6340454]

test_boring_15 = [2110191, 6340440, 10540736, 2030056, 2090141, 2090811, 9470699, 6340453, 10590732, 6340454, 3170245,
                  3170245, 6340454, 6340455, 4240290]

test_exc_5 = [9470663, 1020013, 9470661, 6340802, 2090151]

test_exc_10 = [9470663, 1020013, 9470661, 6340802, 2090151, 2090141, 2090150, 9480686, 2090151, 5270325]

test_exc_15 = [9470663, 1020013, 9470661, 6340802, 2090151, 2090141, 2090150, 9480686, 2090151, 5270325, 9460659,
               9470661, 9470661, 9460656, 10540736]

print('simple binary')
arr = [10203001, 10203002, 10203001, 10203002]
arr_uniq, arr_counts = np.unique(arr, return_counts=True)
arr_probs = arr_counts / len(arr)
print(arr_probs)
simple = stirling_div_uniq(np.array(arr_uniq), np.array(arr_probs))
print(simple)
print('shannon', entropy(arr_probs))
print('')

print('simple quarternary')
arr = [10203001, 10203002, 10203003, 10203004]
arr_uniq, arr_counts = np.unique(arr, return_counts=True)
arr_probs = arr_counts / len(arr)
print(arr_probs)
simple = stirling_div_uniq(np.array(arr_uniq), np.array(arr_probs))
print(simple)
print('shannon', entropy(arr_probs))
print('')

print('simple quarternary long balanced')
arr = [10203001, 10203002, 10203003, 10203004, 10203001, 10203002, 10203003, 10203004]
arr_uniq, arr_counts = np.unique(arr, return_counts=True)
arr_probs = arr_counts / len(arr)
print(arr_probs)
simple = stirling_div_uniq(np.array(arr_uniq), np.array(arr_probs))
print(simple)
print('shannon', entropy(arr_probs))
print('')

print('simple long biased')
arr = [10203001, 10203002, 10203003, 10203004, 10203001, 10203002, 10203001, 10203002]
arr_uniq, arr_counts = np.unique(arr, return_counts=True)
arr_probs = arr_counts / len(arr)
print(arr_probs)
simple = stirling_div_uniq(np.array(arr_uniq), np.array(arr_probs))
print(simple)
print('shannon', entropy(arr_probs))
print('')

print('no weights')
arr = [10203001, 10203002, 10203003, 10203004, 10203005, 10203006, 10203007, 10203008]
arr_uniq, arr_counts = np.unique(arr, return_counts=True)
arr_probs = arr_counts / len(arr)
print(arr_probs)
simple = stirling_div_uniq(np.array(arr_uniq), np.array(arr_probs))
print(simple)
print('shannon', entropy(arr_probs))
print('')

print('weights')
arr = [10203001, 10203002, 10203003, 10203004, 10303005, 10403006, 20203007, 30203008]
arr_uniq, arr_counts = np.unique(arr, return_counts=True)
arr_probs = arr_counts / len(arr)
print(arr_probs)
simple = stirling_div_uniq(np.array(arr_uniq), np.array(arr_probs))
print(simple)
print('shannon', entropy(arr_probs))
print('')

print('boring')
print('++++++')
arr_uniq, arr_counts = np.unique(boring_arr, return_counts=True)
arr_probs = arr_counts / len(boring_arr)
print('len arr: ', len(boring_arr))
print('len uniq: ', len(arr_uniq))
result = stirling_div_uniq(np.array(arr_uniq), np.array(arr_probs))
print(result)
result = stirling_div_mod(np.array(arr_uniq), np.array(arr_probs))
print('alt', result)
print('sum', np.sum(arr_probs))
print('shannon', entropy(arr_probs))
print('')

print('exciting 41')
print('++++++++')
arr_uniq, arr_counts = np.unique(exciting_arr[:41], return_counts=True)
arr_probs = arr_counts / len(exciting_arr[:41])
print('len arr: ', len(exciting_arr[:41]))
print('len uniq: ', len(arr_uniq))
print('test len uniq', len(set(exciting_arr[:41])))
result = stirling_div_uniq(np.array(arr_uniq), np.array(arr_probs))
print(result)
result = stirling_div_mod(np.array(arr_uniq), np.array(arr_probs))
print('alt', result)
print('sum', np.sum(arr_probs))
print('shannon', entropy(arr_probs))
print('')

# TODO - this is wrong
print('exciting')
print('++++++++')
arr_uniq, arr_counts = np.unique(exciting_arr, return_counts=True)
arr_probs = arr_counts / len(exciting_arr)
print('len arr: ', len(exciting_arr))
print('len uniq: ', len(arr_uniq))
print('test len uniq', len(set(exciting_arr)))
result = stirling_div_uniq(np.array(arr_uniq), np.array(arr_probs))
print(result)
result = stirling_div_mod(np.array(arr_uniq), np.array(arr_probs))
print('alt', result)
print('sum', np.sum(arr_probs))
print('shannon', entropy(arr_probs))
print('')
