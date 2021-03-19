# Warwick and Clarke taxonomic distance - Clarke & Warwick: Weighting step lengths for taxonomic distinctness
# note that this doesn't necessarily add much benefit to topological distance
# also that it constrains the measure to the specific index

# percentage decrease, counting as k+1 for k to k step
# e.g. for [395, 170, 39, 7, 4, 2]
# e.g. first step is 1 - (170 / 395), last step is 1 - (1/2)
# calculations:
# [0.57, 0.771, 0.821, 0.429, 0.5, 0.5]
# additive:
# [0.57, 1.34, 2.161, 2.589, 3.089, 3.589]
# scaled:
# [15.87, 37.339, 60.199, 72.139, 86.07, 100.0]

import numpy as np

#s = [395, 170, 39, 7, 4, 2]
s = [618, 52, 9]
print('in', s)
s = np.array(s)

additive = []
for i in range(len(s)):
    if i != len(s) - 1:
        additive.append(1 - (s[i+1] / s[i]))
    else:
        additive.append(1 - (1 / s[i]))

print('out', [round(float(a), 3) for a in additive])
additive = np.array(additive)

for i in range(len(additive)):
    if i != 0:
        additive[i] = additive[i-1] + additive[i]

print('additive', [round(float(a), 3) for a in additive])
additive = np.array(additive)

scaled = additive / additive.max() * 100
print('scaled', [round(float(o), 3) for o in scaled.flatten()])



