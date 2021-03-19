import numba
import numpy as np


# don't track master predecessor and distances matrices - this would be memory prohibitive
# i.e. calculate centralities as you go
# provide arrays of data for each vert - nbs, lens, angles -> the maximum cardinality on database table is 8
# provide 1d arrays for storing incremental data


@numba.njit
def crow_flies(source_x, source_y, max_dist, x_arr, y_arr):
    # filter by distance
    total_count = len(x_arr)
    crow_flies = np.full(total_count, False)
    trim_count = 0
    for i in range(total_count):
        dist = np.sqrt((x_arr[i] - source_x) ** 2 + (y_arr[i] - source_y) ** 2)
        if dist <= max_dist:
            crow_flies[i] = True
            trim_count += 1

    # populate the trimmed to full index map, also populate a reverse index for mapping the neighbours
    trim_to_full_idx_map = np.full(trim_count, np.nan)
    full_to_trim_idx_map = np.full(total_count, np.nan)
    counter = 0
    for i in range(total_count):
        if crow_flies[i]:
            trim_to_full_idx_map[counter] = i
            full_to_trim_idx_map[i] = counter
            counter += 1

    return trim_count, trim_to_full_idx_map, full_to_trim_idx_map


@numba.njit
def graph_window(source_idx, max_dist, x_arr, y_arr, nbs_arr, lens_arr):
    # filter by distance
    source_x = x_arr[source_idx]
    source_y = y_arr[source_idx]
    trim_count, trim_to_full_idx_map, full_to_trim_idx_map = crow_flies(source_x, source_y, max_dist, x_arr, y_arr)

    # trimmed versions of network
    nbs_trim = np.full((trim_count, nbs_arr.shape[1]), np.nan)
    lens_trim = np.full((trim_count, nbs_arr.shape[1]), np.nan)
    # populate
    for i, original_idx in enumerate(trim_to_full_idx_map):
        # using count instead of enumerate because some neighbours are nan
        # this can interfere with the centrality path algorithm which breaks the loop when encountering nans
        j = 0
        # don't confuse j and n!!!
        for n, nb in enumerate(nbs_arr[np.int(original_idx)]):
            # break once all neighbours processed
            if np.isnan(nb):
                break
            # map the original neighbour index to the trimmed version
            nb_trim_idx = full_to_trim_idx_map[np.int(nb)]
            # some of the neighbours will exceed crow flies distance, in which case they have no mapping
            if np.isnan(nb_trim_idx):
                continue
            nbs_trim[i][j] = nb_trim_idx
            # lens and angles can be transferred directly
            lens_trim[i][j] = lens_arr[np.int(original_idx)][n]
            j += 1

    # setup centrality path arrays
    active = np.full(trim_count, np.nan)
    dist_map_m = np.full(trim_count, np.inf)
    pred_map_m = np.full(trim_count, np.nan)
    # get trimmed version of source index
    trim_source_idx = np.int(full_to_trim_idx_map[source_idx])

    return trim_source_idx, trim_to_full_idx_map, active, dist_map_m, pred_map_m, nbs_trim, lens_trim


# angular has to be separate, can't overload numba jitted function without causing typing issues
@numba.njit
def graph_window_angular(source_idx, max_dist, x_arr, y_arr, nbs_arr, lens_arr, angles_arr):
    # filter by network
    source_x = x_arr[source_idx]
    source_y = y_arr[source_idx]
    trim_count, trim_to_full_idx_map, full_to_trim_idx_map = crow_flies(source_x, source_y, max_dist, x_arr, y_arr)

    # trimmed versions of data
    nbs_trim = np.full((trim_count, nbs_arr.shape[1]), np.nan)
    lens_trim = np.full((trim_count, nbs_arr.shape[1]), np.nan)
    angles_trim = np.full((trim_count, nbs_arr.shape[1]), np.nan)
    # populate
    for i, original_idx in enumerate(trim_to_full_idx_map):
        # using count instead of enumerate because some neighbours are nan
        # this can interfere with the centrality path algorithm which breaks the loop when encountering nans
        j = 0
        # don't confuse j and n!!!
        for n, nb in enumerate(nbs_arr[np.int(original_idx)]):
            # break once all neighbours processed
            if np.isnan(nb):
                break
            # map the original neighbour index to the trimmed version
            nb_trim_idx = full_to_trim_idx_map[np.int(nb)]
            # some of the neighbours will exceed crow flies distance, in which case they have no mapping
            if np.isnan(nb_trim_idx):
                continue
            nbs_trim[i][j] = nb_trim_idx
            # lens and angles can be transferred directly
            lens_trim[i][j] = lens_arr[np.int(original_idx)][n]
            angles_trim[i][j] = angles_arr[np.int(original_idx)][n]
            j += 1

    # setup centrality path arrays
    active = np.full(trim_count, np.nan)
    dist_map_m = np.full(trim_count, np.inf)
    pred_map_m = np.full(trim_count, np.nan)
    dist_map_a = np.full(trim_count, np.inf)
    dist_map_a_m = np.full(trim_count, np.inf)
    pred_map_a = np.full(trim_count, np.nan)
    # get trimmed version of source index
    trim_source_idx = np.int(full_to_trim_idx_map[source_idx])

    return trim_source_idx, trim_to_full_idx_map, active, dist_map_m, pred_map_m, nbs_trim, lens_trim, \
           dist_map_a, dist_map_a_m, pred_map_a, angles_trim


# parallel and fastmath don't apply
@numba.njit
def shortest_path_tree(nbs_arr, dist_arr, source_idx, active, dist_map_m, pred_map_m, max_dist):
    '''
    This is the no-frills all centrality paths to max dist from source vertex
    '''

    # set starting node
    dist_map_m[source_idx] = 0
    active[source_idx] = source_idx  # store actual index number instead of booleans, easier for iteration below:

    # search to max distance threshold to determine reachable verts
    while np.any(np.isfinite(active)):
        # get the index for the min of currently active vert distances
        # note, this index corresponds only to currently active vertices
        # min_idx = np.argmin(dist_map_m[np.isfinite(active)])
        # map the min index back to the vertices array to get the corresponding vert idx
        # v = active[np.isfinite(active)][min_idx]
        # v_idx = np.int(v)  # cast to int
        # manually iterating definitely faster
        min_idx = None
        min_dist = np.inf
        for i, d in enumerate(dist_map_m):
            if d < min_dist and np.isfinite(active[i]):
                min_dist = d
                min_idx = i
        v_idx = np.int(min_idx)  # cast to int
        # set current vertex to visited
        active[v_idx] = np.inf
        # visit neighbours
        # for n, dist in zip(nbs_arr[v_idx], dist_arr[v_idx]):
        # manually iterating a tad faster
        for i, n in enumerate(nbs_arr[v_idx]):
            # exit once all neighbours visited
            if np.isnan(n):
                break
            n_idx = np.int(n)  # cast to int for indexing
            # distance is previous distance plus new distance
            dist = dist_arr[v_idx][i]
            d = dist_map_m[v_idx] + dist
            # only pursue if less than max and less than prior assigned distances
            if d <= max_dist and d < dist_map_m[n_idx]:
                dist_map_m[n_idx] = d
                pred_map_m[n_idx] = v_idx
                active[n_idx] = n_idx  # using actual vert idx instead of boolean to simplify finding indices


# parallel and fastmath don't apply (fastmath causes issues...)
# if using np.inf for max-dist, then handle excessive distance issues from callee function
@numba.njit
def shortest_path_tree_angular(nbs_arr, ang_dist_arr, dist_arr, source_idx, dist_map_m, active, dist_map_a,
                               dist_map_a_m, pred_map_a, max_dist):
    '''
    This is the angular variant which has more complexity:
    - using source and target version of dijkstra because there are situations where angular routes exceed max dist
    - i.e. algo searches for and quits once target reached
    - returns both angular and corresponding euclidean distances
    - checks that centrality path algorithm doesn't back-step
    '''

    # set starting node
    dist_map_a[source_idx] = 0
    dist_map_a_m[source_idx] = 0
    active[source_idx] = source_idx  # store actual index number instead of booleans, easier for iteration below:

    # search to max distance threshold to determine reachable verts
    while np.any(np.isfinite(active)):
        # get the index for the min of currently active vert distances
        # note, this index corresponds only to currently active vertices
        # min_idx = np.argmin(dist_map_a[np.isfinite(active)])
        # map the min index back to the vertices array to get the corresponding vert idx
        # v = active[np.isfinite(active)][min_idx]
        # v_idx = np.int(v)  # cast to int
        # manually iterating definitely faster
        min_idx = None
        min_ang = np.inf
        for i, a in enumerate(dist_map_a):
            if a < min_ang and np.isfinite(active[i]):
                min_ang = a
                min_idx = i
        v_idx = np.int(min_idx)  # cast to int
        # set current vertex to visited
        active[v_idx] = np.inf
        # visit neighbours
        # for n, degrees, meters in zip(nbs_arr[v_idx], ang_dist_arr[v_idx], dist_arr[v_idx]):
        # manually iterating a tad faster
        for i, n in enumerate(nbs_arr[v_idx]):
            # exit once all neighbours visited
            if np.isnan(n):
                break
            n_idx = np.int(n)  # cast to int for indexing
            # check that the neighbour node does not exceed the euclidean distance threshold
            if dist_map_m[n_idx] > max_dist:
                continue
            # check that the neighbour was not directly accessible from the prior node
            # this prevents angular backtrack-shortcutting
            # first get the previous vert's id
            pred_idx = pred_map_a[v_idx]
            # need to check for nan in case of first vertex
            if np.isfinite(pred_idx):
                # could check that previous is not equal to current neighbour but would automatically die-out...
                # don't proceed with this index if it could've been followed from predecessor
                # if np.any(nbs_arr[np.int(prev_nb_idx)] == n_idx):
                if np.any(nbs_arr[np.int(pred_idx)] == n_idx):
                    continue
            # distance is previous distance plus new distance
            degrees = ang_dist_arr[v_idx][i]
            d_a = dist_map_a[v_idx] + degrees
            meters = dist_arr[v_idx][i]
            d_m = dist_map_a_m[v_idx] + meters
            # only pursue if angular distance is less than prior assigned distance
            if d_a < dist_map_a[n_idx]:
                dist_map_a[n_idx] = d_a
                dist_map_a_m[n_idx] = d_m
                pred_map_a[n_idx] = v_idx
                active[n_idx] = n_idx


@numba.njit
def assign_accessibility_data(network_x_arr, network_y_arr, data_x_arr, data_y_arr, max_dist):
    '''
    assign data from an x, y array of data point (e.g. landuses)
    to the nearest corresponding point on an x, y array from a network

    This is done once for the whole graph because it only requires a one-dimensional array

    i.e. the similar crow-flies step for the network graph windowing has to happen inside the nested iteration
    because it would require an NxM matrix if done here - which is memory prohibitive
    '''

    verts_count = len(network_x_arr)
    data_count = len(data_x_arr)
    # prepare the arrays for tracking the respective nearest vertex
    data_assign_map = np.full(data_count, np.nan)
    # and the corresponding distance
    data_assign_dist = np.full(data_count, np.nan)
    # iterate each data point
    for data_idx in range(data_count):
        # iterate each network id
        for network_idx in range(verts_count):
            # get the distance
            dist = np.sqrt((network_x_arr[network_idx] - data_x_arr[data_idx]) ** 2 + (
                    network_y_arr[network_idx] - data_y_arr[data_idx]) ** 2)
            # only proceed if it is less than the max dist cutoff
            if dist > max_dist:
                continue
            # if within the cutoff distance
            # and if no adjacent network point has yet been assigned for this data point
            # then proceed to record this adjacency and the corresponding distance
            elif np.isnan(data_assign_dist[data_idx]):
                data_assign_dist[data_idx] = dist
                data_assign_map[data_idx] = network_idx
            # otherwise, only update if the new distance is less than any prior distances
            elif dist < data_assign_dist[data_idx]:
                data_assign_dist[data_idx] = dist
                data_assign_map[data_idx] = network_idx

    return data_assign_map, data_assign_dist


@numba.njit
def accessibility_agg(source_idx, max_dist, nbs, lens, data_classes,
                      network_x_arr, network_y_arr, data_x_arr, data_y_arr, data_assign_map, data_assign_dist):
    # window the network graph
    # this is done for each visited node instead of once on an NxN matrix, which would be memory probitive
    trim_source_idx, trim_to_full_idx_map, active, dist_map, pred_map, nbs_trim, lens_trim = \
        graph_window(source_idx, max_dist, network_x_arr, network_y_arr, nbs, lens)

    # generate network distances
    # using default max_dist of np.inf so that prior nodes can be investigated in below loops
    # still constrained by graph_window above
    shortest_path_tree(nbs_trim, lens_trim, trim_source_idx, active, dist_map, pred_map, max_dist=np.inf)

    # now window the data
    source_x = network_x_arr[source_idx]
    source_y = network_y_arr[source_idx]
    data_trim_count, data_trim_to_full_idx_map, data_full_to_trim_idx_map = \
        crow_flies(source_x, source_y, max_dist, data_x_arr, data_y_arr)

    # iterate the distance trimmed data point
    data_classes_trim = np.full(data_trim_count, np.nan)
    data_distances_trim = np.full(data_trim_count, np.inf)
    for i, original_data_idx in enumerate(data_trim_to_full_idx_map):
        # find the network node that it was assigned to
        assigned_network_idx = data_assign_map[np.int(original_data_idx)]
        # now iterate the trimmed network distances
        for j, (original_network_idx, dist) in enumerate(zip(trim_to_full_idx_map, dist_map)):
            # no need to continue if it doesn't match the data point's assigned network node idx
            if original_network_idx != assigned_network_idx:
                continue
            # check both current and previous nodes for valid distances before continuing
            # first calculate the distance to the current node
            # in many cases, dist_calc will be np.inf, though still works for the logic below
            dist_calc = dist + data_assign_dist[np.int(original_data_idx)]
            # get the predecessor node so that distance to prior node can be compared
            # in some cases this is closer and therefore use the closer corner, especially for full networks
            prev_netw_node_trim_idx = pred_map[j]
            # some will be unreachable causing dist = np.inf or prev_netw_node_trim_idx = np.nan
            if not np.isfinite(prev_netw_node_trim_idx):
                # in this cases, just check whether dist_calc is less than max and continue
                if dist_calc <= max_dist:
                    data_distances_trim[i] = dist_calc
                    data_classes_trim[i] = data_classes[np.int(original_data_idx)]
                continue
            # otherwise, go-ahead and calculate for the prior node
            prev_netw_node_full_idx = np.int(trim_to_full_idx_map[np.int(prev_netw_node_trim_idx)])
            prev_dist_calc = dist_map[np.int(prev_netw_node_trim_idx)] + \
                             np.sqrt(
                                 (network_x_arr[prev_netw_node_full_idx] - data_x_arr[np.int(original_data_idx)]) ** 2
                                 + (network_y_arr[prev_netw_node_full_idx] - data_y_arr[
                                     np.int(original_data_idx)]) ** 2)
            # use the shorter distance between the current and prior nodes
            # but only if less than the maximum distance
            if dist_calc < prev_dist_calc and dist_calc <= max_dist:
                data_distances_trim[i] = dist_calc
                data_classes_trim[i] = data_classes[np.int(original_data_idx)]
            elif prev_dist_calc < dist_calc and prev_dist_calc <= max_dist:
                data_distances_trim[i] = prev_dist_calc
                data_classes_trim[i] = data_classes[np.int(original_data_idx)]

    # note that some entries will be nan values if the max distance was exceeded
    return data_classes_trim, data_distances_trim


@numba.njit
def accessibility_agg_angular(source_idx, max_dist, nbs, lens, angles,
                              data_classes, network_x_arr, network_y_arr, data_x_arr, data_y_arr, data_assign_map,
                              data_assign_dist):
    # window the network graph
    # this is done for each visited node instead of once on an NxN matrix, which would be memory probitive
    trim_source_idx, trim_to_full_idx_map, active, dist_map_m, pred_map_m, nbs_trim, lens_trim, \
    dist_map_a, dist_map_a_m, pred_map_a, angles_trim = \
        graph_window_angular(source_idx, max_dist, network_x_arr, network_y_arr, nbs, lens, angles)

    # generate network distances
    # using default max_dist of np.inf so that prior nodes can be investigated in below loops
    # still constrained by graph_window above
    shortest_path_tree_angular(nbs_trim, angles_trim, lens_trim, trim_source_idx, dist_map_m, active, dist_map_a,
                               dist_map_a_m, pred_map_a, max_dist=np.inf)
    # only dist_map_a_m, pred_map_a used from hereon

    # now window the data
    source_x = network_x_arr[source_idx]
    source_y = network_y_arr[source_idx]
    data_trim_count, data_trim_to_full_idx_map, data_full_to_trim_idx_map = \
        crow_flies(source_x, source_y, max_dist, data_x_arr, data_y_arr)

    # iterate the distance trimmed data point
    data_classes_trim = np.full(data_trim_count, np.nan)
    data_distances_trim = np.full(data_trim_count, np.inf)
    for i, original_data_idx in enumerate(data_trim_to_full_idx_map):
        # find the network node that it was assigned to
        assigned_network_idx = data_assign_map[np.int(original_data_idx)]
        # now iterate the trimmed network distances
        # use the angular route (simplest paths) version of distances
        for j, (original_network_idx, dist) in enumerate(zip(trim_to_full_idx_map, dist_map_a_m)):
            # no need to continue if it doesn't match the data point's assigned network node idx
            if original_network_idx != assigned_network_idx:
                continue
            # check both current and previous nodes for valid distances before continuing
            # first calculate the distance to the current node
            # in many cases, dist_calc will be np.inf, though still works for the logic below
            dist_calc = dist + data_assign_dist[np.int(original_data_idx)]
            # get the predecessor node so that distance to prior node can be compared
            # in some cases this is closer and therefore use the closer corner, especially for full networks
            prev_netw_node_trim_idx = pred_map_a[j]
            # some will be unreachable causing dist = np.inf or prev_netw_node_trim_idx = np.nan
            if not np.isfinite(prev_netw_node_trim_idx):
                # in this cases, just check whether dist_calc is less than max and continue
                if dist_calc <= max_dist:
                    data_distances_trim[i] = dist_calc
                    data_classes_trim[i] = data_classes[np.int(original_data_idx)]
                continue
            # otherwise, go-ahead and calculate for the prior node
            prev_netw_node_full_idx = np.int(trim_to_full_idx_map[np.int(prev_netw_node_trim_idx)])
            prev_dist_calc = dist_map_a_m[np.int(prev_netw_node_trim_idx)] + \
                             np.sqrt(
                                 (network_x_arr[prev_netw_node_full_idx] - data_x_arr[np.int(original_data_idx)]) ** 2
                                 + (network_y_arr[prev_netw_node_full_idx] - data_y_arr[
                                     np.int(original_data_idx)]) ** 2)
            # use the shorter distance between the current and prior nodes
            # but only if less than the maximum distance
            if dist_calc < prev_dist_calc and dist_calc <= max_dist:
                data_distances_trim[i] = dist_calc
                data_classes_trim[i] = data_classes[np.int(original_data_idx)]
            elif prev_dist_calc < dist_calc and prev_dist_calc <= max_dist:
                data_distances_trim[i] = prev_dist_calc
                data_classes_trim[i] = data_classes[np.int(original_data_idx)]

    # note that some entries will be nan values if the max distance was exceeded
    return data_classes_trim, data_distances_trim


"""
MAX - TRIMMED version:
======================
This version is better:
- Single core, but better speed, can run several side by side
- Doesn't use fastmath - which is fickle with checking for infinity
- Has a working progress indicator!! (Doesn't work for parallel)

For city-ids:

900 = 0:00:03

0.0, 0.0, 2.0, 4.0, 18.0, 40.0, 140.0, 200.0, 354.0, 498.0, 0.0, 0.0, 2.0, 4.0, 19.0, 44.0, 152.0, 203.0, 299.0, 366.0, 0, 3, 4, 6, 11, 19, 24, 30, 48, 62, 0.0, 229.92546712698766, 371.78504833368504, 756.5172601485627, 2036.8277185349182, 4923.599265484808, 7281.739031515444, 11598.613791357533, 29842.727729616276, 50087.34019690506, 0.0, 178.0, 179.0, 268.0, 824.0, 2056.0, 2997.0, 4555.0, 10758.0, 14528.0, 0.0, 229.92546712698766, 371.78504833368504, 756.5172601485627, 2036.8277185349182, 4923.599265484808, 7281.739031515444, 11598.613791357533, 30398.902621337806, 50643.5150886266)

500 = 0:00:03

0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 4, 7, 9, 17, 37, 62, 96, 165, 263, 0.0, 289.7837178972635, 684.3778446384364, 1026.7490734596117, 2956.9961016185916, 9858.452484191217, 22021.402327354863, 46201.42835447703, 115703.41656545052, 255421.09020387358, 0.0, 242.0, 603.0, 798.0, 2261.0, 7128.0, 15152.0, 27455.0, 55900.0, 103631.0, 0.0, 289.7837178972635, 684.3778446384364, 1026.7490734596117, 2965.077646913216, 9902.396892822508, 23082.45843780224, 49101.7771783445, 119625.67023153997, 260019.82209523238)

100 = 0:00:07

0.0, 0.0, 6.0, 20.0, 66.0, 178.0, 1040.0, 2784.0, 8672.0, 18350.0, 0.0, 0.0, 6.0, 20.0, 75.0, 225.0, 1532.0, 4062.0, 16311.0, 41867.0, 1, 6, 8, 13, 25, 58, 128, 204, 480, 784, 46.7939170320637, 486.4596246283252, 735.4818967918744, 1632.8681277116193, 4804.330576059429, 16728.961823086618, 52387.139061375114, 106545.12650654798, 383374.6393833538, 808038.9871543432, 2.0, 276.0, 382.0, 800.0, 2145.0, 7317.0, 21358.0, 40253.0, 122811.0, 253018.0, 46.7939170320637, 486.4596246283252, 735.4818967918744, 1632.8681277116193, 4804.330576059429, 17536.409291691267, 55857.20549652646, 114651.41577359509, 426416.0820364272, 899081.1381290911)

10 = 0:00:37

0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 2, 2, 4, 8, 11, 22, 36, 62, 152, 49.47651119551237, 124.9776473965149, 124.9776473965149, 446.71201186047614, 1580.0663555560216, 2597.3868400367755, 8389.319020778943, 18159.74079957935, 45215.73129592777, 172841.15101954484, 96.0, 142.0, 142.0, 606.0, 1871.0, 2772.0, 6807.0, 13388.0, 27508.0, 85522.0, 49.47651119551237, 124.9776473965149, 124.9776473965149, 446.71201186047614, 1580.0663555560216, 2597.3868400367755, 9069.632619477956, 20048.66499924182, 53358.59195180485, 199458.30419033044)

MAX - PARALLEL version
======================

900 = 0:00:11

0.0, 0.0, 2.0, 4.0, 18.0, 40.0, 140.0, 200.0, 354.0, 497.0, 0.0, 0.0, 2.0, 4.0, 19.0, 44.0, 152.0, 203.0, 299.0, 366.0, 0, 3, 4, 6, 11, 19, 24, 30, 48, 62, 0.0, 229.92546712698766, 371.78504833368504, 756.5172601485627, 2036.8277185349182, 4923.599265484808, 7281.739031515444, 11598.613791357533, 29842.727729616276, 50087.34019690506, 0.0, 178.0, 179.0, 268.0, 824.0, 2056.0, 2997.0, 4555.0, 10758.0, 14528.0, 0.0, 229.92546712698766, 371.78504833368504, 756.5172601485627, 2036.8277185349182, 4923.599265484808, 7281.739031515444, 11598.613791357533, 30398.902621337806, 50643.5150886266)

500 = 0:00:08

0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 4, 7, 9, 17, 37, 62, 96, 165, 263, 0.0, 289.7837178972635, 684.3778446384364, 1026.7490734596117, 2956.9961016185916, 9858.452484191217, 22021.402327354863, 46201.42835447703, 115703.41656545052, 255421.09020387358, 0.0, 242.0, 603.0, 798.0, 2261.0, 7128.0, 15152.0, 27455.0, 55900.0, 103631.0, 0.0, 289.7837178972635, 684.3778446384364, 1026.7490734596117, 2965.077646913216, 9902.396892822508, 23082.45843780224, 49101.7771783445, 119625.67023153997, 260019.82209523238)

100 = 0:00:08

0.0, 0.0, 6.0, 20.0, 66.0, 178.0, 1040.0, 2784.0, 8672.0, 18350.0, 0.0, 0.0, 6.0, 20.0, 75.0, 225.0, 1532.0, 4062.0, 16311.0, 41866.0, 1, 6, 8, 13, 25, 58, 128, 204, 480, 784, 46.7939170320637, 486.4596246283252, 735.4818967918744, 1632.8681277116193, 4804.330576059429, 16728.961823086618, 52387.139061375114, 106545.12650654798, 383374.6393833538, 808038.9871543432, 2.0, 276.0, 382.0, 800.0, 2145.0, 7317.0, 21358.0, 40253.0, 122811.0, 253018.0, 46.7939170320637, 486.4596246283252, 735.4818967918744, 1632.8681277116193, 4804.330576059429, 17536.409291691267, 55857.20549652646, 114651.41577359509, 426416.0820364272, 899081.1381290911)

10 = 0:01:18

0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 2, 2, 4, 8, 11, 22, 36, 62, 152, 49.47651119551237, 124.9776473965149, 124.9776473965149, 446.71201186047614, 1580.0663555560216, 2597.3868400367755, 8389.319020778943, 18159.74079957935, 45215.73129592777, 172841.15101954484, 96.0, 142.0, 142.0, 606.0, 1871.0, 2772.0, 6807.0, 13388.0, 27508.0, 85522.0, 49.47651119551237, 124.9776473965149, 124.9776473965149, 446.71201186047614, 1580.0663555560216, 2597.3868400367755, 9069.632619477956, 20048.66499924182, 53358.59195180485, 199458.30419033044)

OLD NOTES
=========

# as tested on full graph for city 500 - based on actual algo time
# 1 max = ~0:00:19 with iterative masking... slow
# 1 max = ~0:00:08 without iterative masking...
# 1 max = ~0:00:04 without angular
# 1 max = ~0:00:06 with manual iteration for metric instead of masking...
# 1 max = ~0:00:05 with manual iteration for metric and angular instead of masking...
# 1 max = ~0:00:04 after some optimisations...
# 1 max = ~0:00:09 with outer prange - not as much speed up as larger city
# 1 max no jit = 0:00:32
# 2 max = 0:00:12
# max w trim ~0:00:03

# as tested on full graph for city 100 - durations at different max search distances
# 1 max = 0:01:49
# 2 max = 0:02:19
# 3 max = 0:02:28
# 5 max = 0:02:31  /  Duration: 0:02:35
# 5 max with internal arrays = 0:02:36
# 5 max with fill instead of index 0:02:27
# 5 max with internal arrays and fill instead of index 0:02:29
# no max = 0:02:33 - max dist searched: 7660 = 4.78

# as tested on full graph for city 100 - based on actual algo time
# 1 max = ~0:00:15 (after optimisations)
# 1 max = ~0:00:08 upper PRANGE
# max w trim 0:00:07
# no max = ~0:00:55

# as tested on full graph for city 10 - based on actual algo time
# 1 max = ~0:06:51
# 1 max = ~0:05:44 PRANGE
# 1 max = ~0:06:10 PRANGE JUST FINITE
# 1 max = ~0:07:48 ??? PRANGE version... perhaps because CPU is hot, slightly throttled
# 1 max = ~0:07:48 ??? PRANGE version... nogil removed
# 1 max = ~0:07:39 ??? Non PRANGE version but still parallelised
# 1 max = ~0:02:17 !!! upper PRANGE version!
# 1 max = ~0:01:55 !!! checking upper PRANGE version (more CPUs available)
# 1 max = ~0:03:39 !!! trying upper PRANGE but back with
# 1 max = ~0:01:16 with outer prange and fastmath
# max with trim 0:00:34

# as tested on full graph for city 5 - based on actual algo time:
# 1 max = ~0:11:05 after parallel improvements and fastmath

# as tested on full graph for city 2 - based on actual algo time:
# 1 max = ~1:36:06

# full network - city 1
# 1 max with PRANGE 13:09:46
# 1 max with trim !!!! 0:12:48

# 100m network - city 1
# 1 max with trim !!!! 0:25:52

# 50m network - city 1
# 1 max with trim !!!! 1:21:50

# 20m network - city 1
# 1 max with trim !!!!

"""
