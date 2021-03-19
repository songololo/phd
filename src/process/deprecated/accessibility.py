'''
For computing accessibilities

'''

import numba
import numpy as np


# os.environ['NUMBA_WARNINGS'] = '1'
# os.environ['NUMBA_DEBUG_ARRAY_OPT_STATS'] = '1'


@numba.njit
def accessibility_os_poi(classes, distances, beta):
    if not beta < 0:
        raise ValueError('NOTE -> aborting: beta should be in the negative form')

    accommodation = 0
    eating = 0
    commercial = 0
    cultural = 0
    tourism = 0
    sports = 0
    entertainment = 0
    health = 0
    education = 0
    manufacturing = 0
    retail = 0
    transport = 0
    parks = 0

    for i in numba.prange(len(classes)):

        cl = classes[i]
        wt = np.exp(distances[i] * beta)

        if cl >= 1010000 and cl < 1020000:
            accommodation += wt
        elif cl >= 1020000 and cl < 2000000:
            eating += wt
        elif cl >= 2000000 and cl < 3000000:
            commercial += wt
        elif cl >= 3170000 and cl < 3180000:
            cultural += wt
        elif cl >= 3200000 and cl < 4000000:
            tourism += wt
        elif cl >= 4000000 and cl < 4250000:
            sports += wt
        elif cl >= 4250000 and cl < 5000000:
            entertainment += wt
        elif cl >= 5270000 and cl < 5310000:
            health += wt
        elif cl >= 5310000 and cl < 6000000:
            education += wt
        elif cl >= 7000000 and cl < 8000000:
            manufacturing += wt
        elif cl >= 9000000 and cl < 10000000:
            retail += wt
        elif cl >= 10000000 and cl < 11000000:
            transport += wt
        elif cl >= 3180000 and cl < 3190000:
            parks += wt

    return accommodation, \
           eating, \
           commercial, \
           tourism, \
           entertainment, \
           manufacturing, \
           retail, \
           transport, \
           health, \
           education, \
           parks, \
           cultural, \
           sports
