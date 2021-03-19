import logging
import itertools
import asyncpg
import numpy as np
import networkx as nx
from shapely import wkb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_networkX_graph(db_config, nodes_table, links_table, boundary_table, city_pop_id):

    logger.info(f'Generating graph from database for {nodes_table} city id: {city_pop_id}')

    # load all vertices and corresponding neighbours and edges
    db_con = await asyncpg.connect(**db_config)
    logger.info('NOTE -> creating temporary buffered city geom')
    # convex hull prevents potential issues with multipolygons deriving from buffer...
    temp_id = await db_con.fetchval(f'''
        INSERT INTO {boundary_table} (geom)
            VALUES ((SELECT ST_ConvexHull(ST_Buffer(geom, 1600))
                FROM {boundary_table} WHERE pop_id = {city_pop_id}))
            RETURNING id;''')

    logger.info('Loading vertices data')
    db_data = await db_con.fetch(f'''
        SELECT nodes.id, nodes.neighbour_nodes, nodes.edges, nodes.city_pop_id, 
                ST_X(nodes.geom) as x, ST_Y(nodes.geom) as y
            FROM {nodes_table} as nodes,
                (SELECT geom FROM {boundary_table} WHERE id = {temp_id}) as boundary
            WHERE ST_Contains(boundary.geom, nodes.geom);''')

    logger.info('NOTE -> removing temporary buffered city geom')
    await db_con.execute(f'DELETE FROM {boundary_table} WHERE id = $1;', temp_id)

    G = nx.Graph()

    # iterate the data the first time to add the nodes
    logger.info('NOTE -> loading nodes')
    for n in tqdm(db_data):
        G.add_node(n['id'],
            x=n['x'],
            y=n['y'],
            live=n['city_pop_id']==city_pop_id)

    # add the edges - but only if both the source and end nodes are loaded
    # this is so that you don't add neighbours beyond buffered extent to graph...
    logger.info('NOTE -> loading edges')
    for n in tqdm(db_data):
        n_id = n['id']
        for nb_id, edge_id in zip(n['neighbour_nodes'], n['edges']):
            # if both the node and its neighbour are in the graph, then add the edge
            if n_id in G and nb_id in G:
                g = await db_con.fetchval(f'''SELECT geom FROM {links_table} WHERE id = $1''', edge_id)
                G.add_edge(n_id, nb_id, geom=wkb.loads(g, hex=True))

    return G


async def load_networkX_graph_angular(db_config, nodes_table, links_table, boundary_table, city_pop_id):

    logger.info(f'Generating dual graph from database for {nodes_table} city id: {city_pop_id}')

    # load all vertices and corresponding neighbours and edges
    db_con = await asyncpg.connect(**db_config)

    # load all vertices and corresponding neighbours and edges
    logger.info('NOTE -> creating temporary buffered city geom')
    # convex hull prevents potential issues with multipolygons deriving from buffer...
    temp_id = await db_con.fetchval(f'''
                INSERT INTO {boundary_table} (geom)
                    VALUES ((SELECT ST_ConvexHull(ST_Buffer(geom, 1600))
                        FROM {boundary_table} WHERE pop_id = {city_pop_id}))
                    RETURNING id;''')

    logger.info('Loading primary vertices data')
    db_data = await db_con.fetch(f'''
                SELECT nodes.id, nodes.edges, nodes.city_pop_id
                    FROM {nodes_table} as nodes,
                        (SELECT geom FROM {boundary_table} WHERE id = {temp_id}) as boundary
                    WHERE ST_Contains(boundary.geom, nodes.geom);''')

    logger.info('NOTE -> removing temporary buffered city geom')
    await db_con.execute(f'''DELETE FROM {boundary_table} WHERE id = {temp_id};''')

    G = nx.Graph()

    # iterate the data the first time to add the nodes
    logger.info('NOTE -> loading dual graph from database')
    for primal_node in tqdm(db_data):

        edges = primal_node['edges']

        # only primary edge pairs (i.e. dual vertices) are here split and combined into a new interlinking dual-edges
        # i.e. primary vertices with a single neighbour / primary edge are automatically ignored
        for source, target in itertools.combinations(edges, 2):
            e_data = await db_con.fetchrow(f'''
                SELECT ST_Length(w.welded_geom) as len, a.angle, s.source_dual_vert_geom, t.target_dual_vert_geom,
                        ST_X(s.source_dual_vert_geom) as source_x, ST_Y(s.source_dual_vert_geom) as source_y,
                        ST_X(t.target_dual_vert_geom) as target_x, ST_Y(t.target_dual_vert_geom) as target_y,
                        ST_MakeLine(s.source_dual_vert_geom, t.target_dual_vert_geom) as dual_simple_edge_geom
                      FROM
                    (SELECT start_node, end_node, geom, ST_Force2D(ST_LineInterpolatePoint(geom, 0.5)) as source_dual_vert_geom
                      FROM {links_table} WHERE id = {source}) as s
                    CROSS JOIN LATERAL
                    (SELECT start_node, end_node, geom, ST_Force2D(ST_LineInterpolatePoint(geom, 0.5)) as target_dual_vert_geom
                      FROM {links_table} WHERE id = {target}) as t
                    CROSS JOIN LATERAL
                    (SELECT CASE
                        WHEN s.start_node = t.start_node
                          THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(s.geom, 0, 0.5), ST_LineSubString(t.geom, 0, 0.5))))
                        WHEN s.start_node = t.end_node
                          THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(s.geom, 0, 0.5), ST_LineSubString(t.geom, 0.5, 1))))
                        WHEN s.end_node = t.start_node
                          THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(s.geom, 0.5, 1), ST_LineSubString(t.geom, 0, 0.5))))
                        WHEN s.end_node = t.end_node
                          THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(s.geom, 0.5, 1), ST_LineSubString(t.geom, 0.5, 1))))
                      END as geom) as g
                    CROSS JOIN LATERAL
                    -- very rarely, a multilinestring is returned, in which cases just created a simplified linestring
                    (SELECT CASE
                      WHEN ST_GeometryType(g.geom) = 'ST_MultiLineString'
                        THEN ST_MakeLine(s.source_dual_vert_geom, t.target_dual_vert_geom)
                      ELSE g.geom
                      END as welded_geom) as w
                    CROSS JOIN LATERAL
                    -- get the angle between the start and end of the line
                    (SELECT
                       degrees(ST_Azimuth(ST_StartPoint(w.welded_geom), ST_LineInterpolatePoint(w.welded_geom, 0.1)))::int as az_a,
                       degrees(ST_Azimuth(ST_LineInterpolatePoint(w.welded_geom, 0.9), ST_EndPoint(w.welded_geom)))::int as az_b) as az
                    CROSS JOIN LATERAL
                    -- reduce the angle in cases where computed through the long direction
                    (SELECT abs((abs(az_b - az_a) + 180) % 360 - 180) as angle) as a;
                ''')

            # add the data to the corresponding new dual vertices (silently updates if already exists)
            if not e_data:
                logger.warning(f'Missing edge data for primal node {primal_node["id"]}, continuing...')
                continue

            # add nodes
            G.add_node(source,
                       x=e_data['source_x'],
                       y=e_data['source_y'],
                       geom=e_data['source_dual_vert_geom'],
                       live=primal_node['city_pop_id'] == city_pop_id)
            G.add_node(target,
                       x=e_data['target_x'],
                       y=e_data['target_y'],
                       geom=e_data['target_dual_vert_geom'],
                       live=primal_node['city_pop_id'] == city_pop_id)

            # create a corresponding dual edge
            s_t = sorted([source, target])
            edge_id = '{0}_{1}'.format(*s_t)
            G.add_edge(source, target,
                       label=edge_id,
                       parent_primal_node=primal_node['id'],
                       live=primal_node['city_pop_id'] == city_pop_id,
                       length=e_data['len'],
                       impedance=e_data['angle'],
                       geom=e_data['dual_simple_edge_geom'])

    return G


async def load_imposed_data(db_config, data_table, data_id_col, data_class_col, boundary_table, city_pop_id, max_dist=None, data_where=None):

    logger.info(f'Loading data from database for {data_table} city id: {city_pop_id}')

    db_con = await asyncpg.connect(**db_config)

    # load all vertices and corresponding neighbours and edges
    logger.info('NOTE -> creating temporary buffered city geom')
    # convex hull prevents potential issues with multipolygons deriving from buffer...
    if not max_dist:
        max_dist = 1600
    temp_id = await db_con.fetchval(f'''
        INSERT INTO {boundary_table} (geom)
            VALUES ((SELECT ST_ConvexHull(ST_Buffer(geom, {max_dist}))
                FROM {boundary_table} WHERE pop_id = {city_pop_id}))
            RETURNING id;''')

    logger.info('Loading data')
    q = f'''
        SELECT data.{data_id_col} as uid, data.{data_class_col} as class_code, ST_X(data.geom) as x, ST_Y(data.geom) as y
            FROM {data_table} as data,
                (SELECT geom FROM {boundary_table} WHERE id = {temp_id}) as boundary
            WHERE ST_Contains(boundary.geom, data.geom)
            ORDER BY data.{data_id_col} ASC
        '''
    if data_where:
        q += f' AND {data_where};'
    else:
        q += ';'
    db_data = await db_con.fetch(q)

    logger.info('NOTE -> removing temporary buffered city geom')
    await db_con.execute(f'''DELETE FROM {boundary_table} WHERE id = {temp_id};''')
    await db_con.close()

    graph_len = len(db_data)
    data_ids = np.full(graph_len, np.nan)
    data_classes = np.full(graph_len, np.nan)  # OS POI data uses integers for classification codes
    data_x = np.full(graph_len, np.nan)
    data_y = np.full(graph_len, np.nan)

    for i, data in enumerate(db_data):
        data_ids[i] = data['uid']
        data_classes[i] = data['class_code']
        data_x[i] = data['x']
        data_y[i] = data['y']

    return graph_len, data_ids, data_classes, data_x, data_y
