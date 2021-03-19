import asyncio
import numpy as np
import asyncpg
from numba.typed import List, Dict
from numba import njit, int64, float64

from cityseer.util import graphs
from phd.process.loaders import postGIS_to_networkX

@njit
def shortest_path_tree_entropy(edge_data: np.ndarray, node_edge_map: Dict, src_idx: int, distances: np.ndarray):
    '''
    Modelled on cityseer.networks.shortest_path_tree
    '''
    # arrays
    n = len(node_edge_map)
    active = List.empty_list(int64)
    tree_dists = np.full(n, np.inf)
    tree_preds = np.full(n, np.nan)
    probs_map = (
        List.empty_list(float64),  # 50m
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64),
        List.empty_list(float64)  # 1600m
    )
    # >>>>>
    # the distances dimension
    # information entropy probabilities
    d_n = len(distances)
    tree_probs = np.full((d_n, n), 0.0)
    for i, d in enumerate(distances):
        tree_probs[i, src_idx] = 1
    max_dist = distances.max()
    # >>>>>
    # setup
    tree_dists[src_idx] = 0
    active.append(src_idx)
    while len(active):
        # find min
        min_nd_idx = None
        min_dist = np.inf
        for idx, nd_idx in enumerate(active):
            if tree_dists[nd_idx] < min_dist:
                min_dist = tree_dists[nd_idx]
                min_nd_idx = nd_idx
        active_nd_idx = int(min_nd_idx)
        active.remove(active_nd_idx)
        # >>>>>
        # keep count of newly discovered neighbours so that new probabilities can be calculated
        new_edge_counter = np.full(d_n, 0)
        edge_reverse_counter = np.full(d_n, 0)
        # >>>>>
        # PASS 1 - count new out edges
        for edge_idx in node_edge_map[active_nd_idx]:
            start_nd, end_nd, seg_len, seg_ang, seg_imp_fact, seg_in_bear, seg_out_bear = edge_data[edge_idx]
            nb_nd_idx = int(end_nd)
            # don't follow self-loops
            if nb_nd_idx == active_nd_idx:
                continue
            # don't visit predecessor nodes
            if nb_nd_idx == tree_preds[active_nd_idx]:
                continue
            # enforce max
            dist = tree_dists[active_nd_idx] + seg_len
            # count new neighbours
            if np.isinf(tree_dists[nb_nd_idx]):
                for i, d in enumerate(distances):
                    if dist <= d:
                        new_edge_counter[i] += 1
                        edge_reverse_counter[i] += 1
        # terminate dead-ends
        for i, d in enumerate(distances):
            if new_edge_counter[i] == 0:
                if tree_probs[i, active_nd_idx] != 0:
                    probs_map[i].append(tree_probs[i, active_nd_idx])
                    tree_probs[i, active_nd_idx] = 0
        # PASS 2 - propogate probabilities
        for edge_idx in node_edge_map[active_nd_idx]:
            start_nd, end_nd, seg_len, seg_ang, seg_imp_fact, seg_in_bear, seg_out_bear = edge_data[edge_idx]
            nb_nd_idx = int(end_nd)
            dist = tree_dists[active_nd_idx] + seg_len
            # don't follow self-loops
            if nb_nd_idx == active_nd_idx:
                continue
            # don't visit predecessor nodes
            if nb_nd_idx == tree_preds[active_nd_idx]:
                continue
            # propogate - do before updating distances
            if np.isinf(tree_dists[nb_nd_idx]):
                for i, d in enumerate(distances):
                    if dist <= d:
                        tree_probs[i, nb_nd_idx] = tree_probs[i, active_nd_idx] / new_edge_counter[i]
                        edge_reverse_counter[i] -= 1
                        if edge_reverse_counter[i] == 0:
                            tree_probs[i, active_nd_idx] = 0
            # update distances and predecessors
            if dist <= max_dist and dist < tree_dists[nb_nd_idx]:
                tree_dists[nb_nd_idx] = dist
                tree_preds[nb_nd_idx] = active_nd_idx
                active.append(nb_nd_idx)

    # calculate information entropy
    info_ent_return = np.full(d_n, 0.0)
    for i in range(d_n):
        for prob in probs_map[i]:
            info_ent_return[i] += prob * np.log2(prob)
        info_ent_return[i] *= -1

    return info_ent_return


@njit
def centrality_info_ent_calc(node_map:np.ndarray, edge_map:np.ndarray, node_edge_map:Dict, distances:np.ndarray):
    '''
    This is modelled after the cityseer.networks.network_centralities method
    '''
    n = len(node_map)
    d_n = len(distances)
    nodes_live = node_map[:, 2]
    info_ent_data = np.full((d_n, n), 0.0)
    # iterate through each vert and calculate the centrality path tree
    for src_idx in range(n):
        # numba no object mode can only handle basic printing
        if src_idx % 10000 == 0:
            print('...progress')
            print(round(src_idx / n * 100, 2))
        # only compute for nodes in current city
        if not nodes_live[src_idx]:
            continue
        # run per node calculations
        info_ent_data_node = shortest_path_tree_entropy(edge_map,
                                                        node_edge_map,
                                                        src_idx,
                                                        distances)
        # aggregate to the main array
        for i in range(len(distances)):
            info_ent_data[i][src_idx] = info_ent_data_node[i]
    return info_ent_data


async def centrality_info_ent(db_config, nodes_table, links_table, city_pop_id, distances):
    logger.info(f'Loading graph for city: {city_pop_id} derived from table: {nodes_table}')
    G = await postGIS_to_networkX(db_config, nodes_table, links_table, city_pop_id)
    logger.info(f'Generating node map and edge map')
    node_labels, node_map, edge_map, node_edge_map = graphs.graph_maps_from_nX(G)
    logger.info('Calculating centrality paths and centralities')
    start = time.localtime()
    info_ent_data = centrality_info_ent_calc(node_map, edge_map, node_edge_map, np.array(distances))
    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')
    logger.info('Aggregating data')
    # iterate the 'i' verts
    bulk_data = []
    for v_idx, v_label in enumerate(node_labels):
        # first check that this is a live node (i.e. within the original city boundary)
        # live status is stored in index 2
        if not node_map[v_idx][2]:
            continue
        # compile node data
        node_data = [v_label]
        inner_data = []
        for i in range(len(distances)):
            inner_data.append(info_ent_data[i][v_idx])
        node_data.append(inner_data)
        bulk_data.append(node_data)
    logger.info('Writing data back to database')
    db_con = await asyncpg.connect(**db_config)
    await db_con.execute(f'ALTER TABLE {nodes_table} ADD COLUMN IF NOT EXISTS c_info_ent real[];')
    await db_con.executemany(f'''
        UPDATE {nodes_table} SET c_info_ent = $2 WHERE id = $1
    ''', bulk_data)
    await db_con.close()


if __name__ == '__main__':

    import time
    import datetime
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }
    nodes_table = 'analysis.nodes_20'
    links_table = 'analysis.links_20'
    distances = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]

    for city_pop_id in range(139, 932):
        start_time = time.localtime()
        logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')
        asyncio.run(centrality_info_ent(db_config,
                                        nodes_table,
                                        links_table,
                                        city_pop_id,
                                        distances))
        logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')
        end_time = time.localtime()
        logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
