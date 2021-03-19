import logging

import asyncpg
import networkx as nx
import numpy as np
from shapely import wkb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def postGIS_to_networkX(db_config, nodes_table, links_table, city_pop_id):
    logger.info(f'Generating graph from database for {nodes_table} city id: {city_pop_id}')
    db_con = await asyncpg.connect(**db_config)
    G = nx.Graph()
    logger.info('Loading vertices data')
    node_data = await db_con.fetch(f'''
        SELECT id, within, geom FROM {nodes_table} WHERE city_pop_id = {city_pop_id}''')
    for n in tqdm(node_data):
        g = wkb.loads(n['geom'], hex=True)
        G.add_node(n['id'],
                   x=g.x,
                   y=g.y,
                   live=n['within'])
    logger.info('Loading edges data')
    edge_data = await db_con.fetch(f'''
            SELECT id, node_a, node_b, geom FROM {links_table} WHERE city_pop_id = {city_pop_id}''')
    await db_con.close()
    for e in tqdm(edge_data):
        G.add_edge(
            e['node_a'],
            e['node_b'],
            geom=wkb.loads(e['geom'], hex=True))

    return G


async def postGIS_to_landuses_dict(db_config,
                                   data_table,
                                   data_id_col,
                                   data_class_col,
                                   boundary_table,
                                   city_pop_id,
                                   max_dist=1600,
                                   data_where=None):
    logger.info(f'Loading data from database for {data_table} city id: {city_pop_id}')
    db_con = await asyncpg.connect(**db_config)
    # load all vertices and corresponding neighbours and edges
    logger.info('NOTE -> creating temporary buffered city geom')
    # convex hull prevents potential issues with multipolygons deriving from buffer...
    temp_id = await db_con.fetchval(f'''
    INSERT INTO {boundary_table} (geom)
        VALUES ((SELECT ST_ConvexHull(ST_Buffer(geom, {max_dist}))
            FROM {boundary_table} WHERE pop_id = {city_pop_id}))
        RETURNING id;''')
    logger.info('Loading data')
    q = f'''
    SELECT data.{data_id_col} as uid, 
        data.{data_class_col} as class_code, 
        ST_X(data.geom) as x, 
        ST_Y(data.geom) as y
        FROM {data_table} as data,
            (SELECT geom FROM {boundary_table} WHERE id = {temp_id}) as boundary
        WHERE ST_Contains(boundary.geom, data.geom)
    '''
    if data_where:
        q += f' AND {data_where} ORDER BY data.{data_id_col} ASC;'
    else:
        q += f'ORDER BY data.{data_id_col} ASC;'
    logger.info(f'Fetching POI data with query: {q}')
    db_data = await db_con.fetch(q)
    logger.info('NOTE -> removing temporary buffered city geom')
    await db_con.execute(f'''DELETE FROM {boundary_table} WHERE id = {temp_id};''')
    await db_con.close()
    data_dict = {}
    for i, data in enumerate(db_data):
        data_dict[str(data['uid'])] = {
            'class': data['class_code'],
            'x': data['x'],
            'y': data['y']
        }
    return data_dict


async def postGIS_to_valuations_dict(db_config,
                                     data_table,
                                     boundary_table,
                                     city_pop_id,
                                     max_dist=1600,
                                     data_where=None):
    logger.info(f'Loading data from database for {data_table} city id: {city_pop_id}')
    db_con = await asyncpg.connect(**db_config)
    # load all vertices and corresponding neighbours and edges
    logger.info('NOTE -> creating temporary buffered city geom')
    # convex hull prevents potential issues with multipolygons deriving from buffer...
    temp_id = await db_con.fetchval(f'''
        INSERT INTO {boundary_table} (geom)
            VALUES ((SELECT ST_ConvexHull(ST_Buffer(geom, {max_dist}))
                FROM {boundary_table} WHERE pop_id = {city_pop_id}))
            RETURNING id;''')
    logger.info('Loading data')
    q = f'''
        SELECT data.uarn, data.total_area, data.adopted_rateable_value, ST_X(data.geom) as x, ST_Y(data.geom) as y
            FROM {data_table} as data,
                (SELECT geom FROM {boundary_table} WHERE id = {temp_id}) as boundary
            WHERE ST_Contains(boundary.geom, data.geom)
        '''
    if data_where:
        q += f' AND {data_where} ORDER BY data.uarn ASC;'
    else:
        q += 'ORDER BY data.uarn ASC;'
    db_data = await db_con.fetch(q)
    logger.info('NOTE -> removing temporary buffered city geom')
    await db_con.execute(f'''DELETE FROM {boundary_table} WHERE id = {temp_id};''')
    await db_con.close()
    data_dict = {}
    for i, data in enumerate(db_data):
        uarn = str(data['uarn'])
        area = float(data['total_area'])
        val = float(data['adopted_rateable_value'])
        rate_per_sm = np.nan
        # calculate rate per sm
        if area:
            rate_per_sm = val / area
        data_dict[uarn] = {
            'area': area,
            'val': val,
            'rate': rate_per_sm,
            'x': data['x'],
            'y': data['y']
        }
    return data_dict
