"""
(See deprecated folder for Graph Tool variant)

Information Entropy based centrality index.
This has to be breadth-first (not depth-first) to evenly weight route choices in an outwards and distance sensitive manner.
Because the network is split into smaller chunks (20m, 50m, 100m) it avoids issues with artefacts due to length or network structure.
This also avoids potential issues where longer segments more rapidly claim route options than shorter segments.

For city 100 = Execution time: 417s for threshold distance 600
"""
import os
import pandas as pd
import numpy as np
from numba import jit
import psycopg2
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: there is an artefact where the end points on a line sometimes tick up in value, can't figure it out
@jit(nopython=True, nogil=True)
def centrality_numba(node_ids, neighbours, distances, active, probability, agg_distance, processed, new_neighbours,
                     threshold_distance, entropy):

    # NOTE -> don't confuse indexes with node ids
    # NOTE -> also remember that node ids are duplicated for each out edge
    # start iterating node id instances to calculate centrality
    for idx, parent_id in enumerate(node_ids):
        if idx and idx % 1000 == 0:
            completion = round((idx / len(node_ids)) * 100, 2)
            print('...', completion, '%')
        # reset active, probability, agg_distances, processed
        active[:] = False
        probability[:] = 0
        agg_distance[:] = 0
        processed[:] = False
        # set parent node's index to active
        active[idx] = True
        # set parent's probability to 1
        probability[idx] = 1
        # recursively iterate - find neighbours, add to active, and keep going till no more active nodes
        while active.any():
            # check all active nodes
            for active_idx in active.nonzero()[0]:
                # set the node to inactive
                active[active_idx] = False
                # claim the node
                processed[active_idx] = True
                # keep count of valid new neighbours
                new_nb = 0
                # reset the neighbours array
                new_neighbours[:] = False
                # iterate neighbours
                for i, nb_id in enumerate(neighbours[active_idx]):
                    if np.isnan(nb_id):
                        break
                    # check whether neighbouring node is inside bounds of SQL query
                    if not node_ids[node_ids == nb_id].any():
                        continue
                    nb_idx = np.where(node_ids == nb_id)
                    # check that instances of this neighbour id aren't already active or haven't already been processed
                    if processed[nb_idx].any() or active[nb_idx].any():
                        continue
                    # calculate the aggregate network distance
                    agg_dist = agg_distance[active_idx] + distances[active_idx][i]
                    # if the threshold is exceeded, continue
                    if agg_dist > threshold_distance:
                        continue
                    # if all checks have been passed, set to active
                    active[nb_idx] = True
                    # add to neighbours count
                    new_nb += 1
                    # keep track of new neighbours
                    new_neighbours[nb_idx] = True
                    # set the aggregate distance
                    agg_distance[nb_idx] = agg_dist
                if new_nb:
                    # set outgoing probability for new neighbours to old probability / number of new neighbours
                    probability[new_neighbours] = probability[active_idx] / new_nb
                    # reset the parent's probability
                    probability[active_idx] = 0
                # if no new neighbours found
                else:
                    # aggregate probabilities (which are supposed to sum to 1) as entropy in centrality array
                    entropy[idx] += probability[active_idx] * np.log2(probability[active_idx])
    # cast entropy to positive (per formula) and return
    return -entropy

def information_centrality(dsn_string, network_schema, nodes_table, links_table, boundary_schema, threshold_distances, boundary_table, city_pop_id):

    # open database connection
    logger.info(f'loading data from {network_schema}.{nodes_table} for city {city_pop_id}')
    with psycopg2.connect(dsn_string) as db_connection:
        db_cursor = db_connection.cursor()
        # setup temporary buffered boundary based on threshold distance - prevents centrality falloff
        logger.info('NOTE -> creating temporary buffered city geom')
        # convex hull prevents potential issues with multipolygons deriving from buffer...
        db_cursor.execute(f'''
            INSERT INTO {boundary_schema}.{boundary_table} (geom)
                VALUES ((SELECT ST_ConvexHull(ST_Buffer(geom, {max(threshold_distances)}))
                    FROM {boundary_schema}.{boundary_table} WHERE pop_id = {city_pop_id}))
                RETURNING id;''')
        temp_id = db_cursor.fetchone()[0]
        # load nodes
        node_query = f'''
          SELECT nodes.id, nodes.neighbour_nodes as neighbours, nodes.edges as out_edges, nodes.geom
            FROM {network_schema}.{nodes_table} as nodes,
              (SELECT geom FROM {boundary_schema}.{boundary_table} WHERE id = {temp_id}) as boundary
            WHERE ST_Contains(boundary.geom, nodes.geom)
        '''
        logger.info(f'query: {node_query}')
        df_nodes = pd.read_sql(
            sql=node_query,
            con=db_connection,
            index_col='id',
            coerce_float=True,
            params=None
        )
        logger.info(f'{len(df_nodes)} node rows loaded')
        # load links
        link_query = f'''
            SELECT links.id, links.start_node, links.end_node, ST_Length(ST_Transform(links.geom, 27700)) as distance
              FROM {network_schema}.{links_table} as links,
                (SELECT geom FROM {boundary_schema}.{boundary_table} WHERE id = {temp_id}) as boundary
              WHERE ST_Contains(boundary.geom, links.geom)
            '''
        logger.info(f'query: {link_query}')
        df_links = pd.read_sql(
            sql=link_query,
            con=db_connection,
            index_col='id',
            coerce_float=True,
            params=None
        )
        logger.info(f'{len(df_links)} link rows loaded')

        logger.info('NOTE -> removing temporary buffered city geom')
        db_cursor.execute(f'''DELETE FROM {boundary_schema}.{boundary_table} WHERE id = {temp_id}''')


    if len(df_nodes) > 0:

        for threshold_distance in threshold_distances:

            logger.info(f'currently processing threshold distance {threshold_distance}')

            start_time = time.time()

            # add a column of zeros to nodes df for computations and resulting centrality
            active = np.zeros(len(df_nodes), dtype=bool)
            probability = np.zeros(len(df_nodes))
            agg_distance = np.zeros(len(df_nodes))
            processed = np.zeros(len(df_nodes), dtype=bool)
            new_neighbour = np.zeros(len(df_nodes), dtype=bool)
            entropy = np.zeros(len(df_nodes))
            # using preset matrix of 5 width as no nodes are likely to have that many neighbours
            distances = np.zeros((len(df_nodes), 10))
            distances[:] = np.nan
            neighbours = np.zeros((len(df_nodes), 10))
            neighbours[:] = np.nan

            # populate the distances and neighbours arrays
            for i, (index, row) in enumerate(df_nodes.iterrows()):
                for j, neighbour in enumerate(row.neighbours):
                    neighbours[i, j] = neighbour
                for j, out_edge in enumerate(row.out_edges):
                    if out_edge in df_links.index:
                        distances[i, j] = df_links.at[out_edge, 'distance']

            logger.info('running information entropy centrality')

            entropy = centrality_numba(np.array(df_nodes.index), neighbours, distances, active, probability, agg_distance, processed, new_neighbour, threshold_distance, entropy)

            end_time = time.time()
            logger.info(f'Execution time: {end_time - start_time}s for threshold distance {threshold_distance}')

            logger.info('writing probabilities back to nodes dataframe')
            df_nodes = df_nodes.assign(entropy=entropy)

            # push data to new column in database
            logger.info('writing centrality results to database')
            with psycopg2.connect(dsn_string) as db_connection:
                with db_connection.cursor() as db_cursor:
                    # create a matching column for centrality results
                    db_cursor.execute(f'''
                        ALTER TABLE {network_schema}.{nodes_table}
                          ADD COLUMN IF NOT EXISTS centrality_entropy_{threshold_distance} real;
                    ''')
                    # set new values
                    for idx in df_nodes.index:
                        db_cursor.execute(f'''
                            UPDATE {network_schema}.{nodes_table}
                                SET centrality_entropy_{threshold_distance} = %s
                                    WHERE id = %s
                        ''', (float(df_nodes.loc[idx, 'entropy']), int(idx)))

if __name__ == '__main__':

    dsn_string = f"dbname=gareth user=gareth host=localhost port=5432"
    db_schema = 'analysis'
    nodes_table = 'roadnodes_100'
    links_table = 'roadlinks_100'
    boundary_schema = 'analysis'
    boundary_table = 'city_boundaries_150'
    threshold_distances = [200, 400, 800, 1600]

    for city_pop_id in range(1, 2):
        logger.info(f'Starting execution of entropy centrality for city id: {city_pop_id} on table '
                    f'{db_schema}.{nodes_table}')
        information_centrality(dsn_string, db_schema, nodes_table, links_table, boundary_schema,
                               threshold_distances, boundary_table, city_pop_id)

'''
NEIGHBOURHOOD CENTRALITY MEASURE -> NODE OUTWARDS INFORMATION ENTROPY
The nodes and adjacent edges are already computed on the database
For example, to select all nodes for city 100 with max 20m segments use a query such as:
    SELECT node_id, edges FROM analysis.roadlinks_20 WHERE city_pop_id = '100'
The distance buffering is now done using geopandas, for example, when iterating all nodes within 300m of node #1:
data_1_300 = data[...]
'''
'''
start_node = None
# Calculates probability of all possible routes from node within cutoff distance
visitedNodes = set()
claimedEdges = set()
branches = []
probabilities = []
# add origin
# TODO: figure out how to add distance threshold
branches.append((start_node, 1))  # add starting node with starting probability
visitedNodes.add(start_node)  # add starting node to visited set
# start iterating through branches...adding and removing as you go
while branches:
    for base_node, probability in branches:  # was node[0] and node[1]
        newVal = 0.0
        newEdges = []
        # find all out-going edges
        out_edges = []
        # select out_edges ids for all edges with a node = base_node
        #
        for edge in out_edges:
            # each edge is a source, target tuple
            if edge not in claimedEdges:
                newVal += 1
                claimedEdges.add(edge)
                newEdges.append(edge)
        if newVal == 0:
            probabilities.append(probability)
        else:
            out_nodes = None
            # get all out_nodes for neighbouring edges:
            #
            # get the node's neighbours by querying the corresponding graph dict key
            for node in out_nodes:
                # add any unvisited nodes plus probability to branches
                if node not in visitedNodes:
                    visitedNodes.add(node)
                    branches.append((node, newVal * probability))
                # if the outgoing node has already been visited, but the corresponding edge is new:
                elif (?base_node, out_node?) in newEdges:
                    # append the probability instead
                    probabilities.append(newVal * probability)
        branches.remove(base_node)
r = 0
if len(probabilities) == 1:
    outProb = 0
else:
    for k in probabilities:
        r += (1 / k) * math.log(1 / k, 2)
    outProb = -r
'''
