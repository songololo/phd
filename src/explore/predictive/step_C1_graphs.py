# %%
'''
STEP 2:  Prepare network geometries etc. for three scenarios utilised in step 3.

select SUM(ST_Length(links.geom))
	FROM (SELECT geom FROM analysis.links_20) as links,
		(SELECT geom FROM analysis.city_boundaries_150 where id = 28705) as bound
	WHERE ST_Contains(bound.geom, links.geom);
gives: 20174.44130493104

once the footpaths have been added the total lengths come to around 24km

# case 2: 24km of 80m grid

# case 3: 24km of fractal
'''
import asyncio

import asyncpg
import networkx as nx
import numpy as np
from cityseer.util import graphs
from process.loaders import postGIS_to_networkX
from shapely import wkt, geometry


async def inner_york_burb(db_config):
    # load york graph
    york_burb = await postGIS_to_networkX(db_config,
                                          'analysis.nodes_20',
                                          'analysis.links_20',
                                          41)
    print('NX graph summary before pruning')
    print(nx.info(york_burb))
    # manually load inner city boundary
    db_con = await asyncpg.connect(**db_config)
    inner_boundary = await db_con.fetchval(f'''
                        select ST_AsText(ST_buffer(geom, 10))
                        from analysis.city_boundaries_150
                        where id = 28705
                    ''')
    inner_boundary = wkt.loads(inner_boundary)
    # filter out graph nodes that are not within the inner boundary
    drop_nodes = []
    for nd_id, nd_data in york_burb.nodes(data=True):
        nd_geom = geometry.Point(nd_data['x'], nd_data['y'])
        if not inner_boundary.contains(nd_geom):
            drop_nodes.append(nd_id)
    york_burb.remove_nodes_from(drop_nodes)
    print('NX graph summary after pruning')
    print(nx.info(york_burb))
    #
    return york_burb


def york_burb():
    '''
    NX graph summary before pruning
    Name:
    Type: Graph
    Number of nodes: 42590
    Number of edges: 43728
    Average degree:   2.0534
    NX graph summary after pruning
    Name:
    Type: Graph
    Number of nodes: 1298
    Number of edges: 1361
    Average degree:   2.0971
    Yorkburb summed lengths: 23704.85033655231
    '''
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }
    # use process loaders to load graph
    york_burb = asyncio.run(inner_york_burb(db_config))
    # sum lengths for reference
    sum_lengths = 0
    for s, e, d in york_burb.edges(data=True):
        sum_lengths += geometry.LineString(d['geom']).length
    print(f'Yorkburb summed lengths: {sum_lengths}')
    # adjust x / y values to smaller coordinate system
    # first pass - find minimums
    min_x = min_y = np.inf
    for n, d in york_burb.nodes(data=True):
        x, y = (d['x'], d['y'])
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
    # second pass - adjust coordinates
    for n in york_burb.nodes():
        old_x = york_burb.nodes[n]['x']
        york_burb.nodes[n]['x'] = old_x - min_x
        old_y = york_burb.nodes[n]['y']
        york_burb.nodes[n]['y'] = old_y - min_y
    # likewise adjust and check geoms
    for s, e, d in york_burb.edges(data=True):
        old_geom = d['geom']
        new_geom = []
        for x, y in old_geom.coords:
            new_geom.append((x - min_x, y - min_y))
        new_geom = geometry.LineString(new_geom)
        d['geom'] = new_geom
        assert old_geom.length == new_geom.length
    # check that total lengths haven't changed
    post_sum_lengths = 0
    for s, e, d in york_burb.edges(data=True):
        post_sum_lengths += geometry.LineString(d['geom']).length
    assert post_sum_lengths == sum_lengths
    # relabel nodes
    rl = {}
    rl_counter = 0
    for n in york_burb.nodes():
        rl[n] = rl_counter
        rl_counter += 1
    york_burb = nx.relabel_nodes(york_burb, rl, copy=True)
    # remove link (shorten dead-end to simplify adding new routes)
    york_burb.remove_edge(1283, 1074)
    # remove node (make way for adjacent edge)
    york_burb.remove_nodes_from([157, 1089, 1144, 1163, 998, 503])
    # add nodes where necessary
    for x, y in [
        (460379.79, 451844.15),
        (460402.13, 451866.82),
        (460429.81, 451876.83),
        (460462.79, 451626.64),
        (460160.19, 451843.77),
        (460147.19, 451864.79),
        (460140.00, 451826.62),
        (460107.08, 451863.13),
        (460160.19, 451797.30),
        (460104.04, 451788.73),
        (460188.12, 451423.61),
        (459840.70, 451434.38),
        (459913.19, 451389.30)]:
        adj_x = x - min_x
        adj_y = y - min_y
        york_burb.add_node(rl_counter, x=adj_x, y=adj_y)
        rl_counter += 1
    # add missing footpaths
    for start_nd, end_nd in [
        (238, 865),
        (1117, 43),
        (797, 67),
        (918, 797),
        (795, 653),
        (365, 797),
        (705, 673),
        (230, 362),
        (1068, 1085),
        (666, 1041),
        (869, 426),
        (116, 991),
        (1097, 991),
        (99, 312),
        (771, 1113),
        (1069, 1218),
        (223, 447),
        (1167, 1186),
        (643, 1049),
        (1034, 185),
        (1189, 886),
        (4, 671),
        (60, 78),
        (359, 1188),
        (540, 1283),
        (1283, 770),
        (770, 817),
        (82, 889),
        (223, 306),
        (874, 304),
        (969, 478),
        (159, 1298),
        (1298, 1299),
        (1299, 1300),
        (1300, 1246),
        (659, 1300),
        (1028, 1299),
        (624, 1298),
        (1298, 616),
        (628, 1303),
        (1303, 1302),
        (1302, 160),
        (1302, 1304),
        (1305, 1304),
        (1304, 1306),
        (945, 1307),
        (1307, 620),
        (1307, 1275),
        (478, 1301),
        (1301, 493),
        (1301, 492),
        (1180, 1308),
        (1308, 856),
        (871, 1308),
        (870, 1309),
        (1309, 1310),
        (1310, 1172),
        (1310, 429),
        (1017, 778)]:
        x_start = (york_burb.nodes[start_nd]['x'])
        y_start = (york_burb.nodes[start_nd]['y'])
        x_end = (york_burb.nodes[end_nd]['x'])
        y_end = (york_burb.nodes[end_nd]['y'])
        geom = geometry.LineString([(x_start, y_start), (x_end, y_end)])
        york_burb.add_edge(start_nd, end_nd, geom=geom)
    #  decompose new edges
    york_burb = graphs.nX_decompose(york_burb, 20)
    # ready
    return york_burb


def grid_ville():
    '''
    Type: Graph
    Number of nodes: 1200
    Number of edges: 1320
    Average degree:   2.2000
    Gridville summed lengths: 24200.000000000062
    '''
    grid_ville = nx.Graph()
    # divisor and extents
    div = 12
    ext = 1100
    # add nodes
    for x_id in range(div):
        for y_id in range(div):
            grid_ville.add_node(f'{x_id}_{y_id}', x=ext / div * x_id, y=ext / div * y_id)
    # add edges
    sum_lengths = 0
    for x_id in range(div):
        for y_id in range(div):
            node_set = []
            # last row and column do not have a next row / column
            # add edge in the x direction
            if y_id < div - 1:
                a_nd_start = f'{x_id}_{y_id}'
                a_nd_end = f'{x_id}_{y_id + 1}'
                node_set.append((a_nd_start, a_nd_end))
            # add edge in the y direction
            if x_id < div - 1:
                b_nd_start = f'{x_id}_{y_id}'
                b_nd_end = f'{x_id + 1}_{y_id}'
                node_set.append((b_nd_start, b_nd_end))
            # for x direction and y direction node sets, add edges and edge geoms
            for start, end in node_set:
                start_x = grid_ville.nodes[start]['x']
                start_y = grid_ville.nodes[start]['y']
                end_x = grid_ville.nodes[end]['x']
                end_y = grid_ville.nodes[end]['y']
                geom = geometry.LineString([(start_x, start_y), (end_x, end_y)])
                grid_ville.add_edge(start, end, geom=geom)
                sum_lengths += geom.length
    #  decompose new edges
    grid_ville = graphs.nX_decompose(grid_ville, 20)
    # print info
    print(nx.info(grid_ville))
    # report sum
    print(f'Gridville summed lengths: {sum_lengths}')
    # ready
    return grid_ville


def suburb():
    '''
    Number of nodes: 1349
    Number of edges: 1348
    Average degree:   1.9985
    Gridville summed lengths: 23550.0. Last length 14.0625
    '''
    suburb = nx.Graph()
    # set params
    recursions = 7
    distance = 1200
    # set the seed centroid
    node_id = 1
    suburb.add_node(node_id, x=distance / 2, y=distance / 2)
    centroids = [node_id]
    node_id += 1
    # sum geom lengths
    sum_lengths = 0
    last_length = np.inf
    # recursively add centroids and edges
    for i in range(recursions):
        # alternate directions
        x_direction = True
        if i % 2 == 0:
            x_direction = False
            # distance only updated every second cycle
            distance = distance / 2 - 25
        # new centroids - keep separate and then replace at end of loop
        new_centroids = []
        # for each centroid
        for start_id in centroids:
            x_start = suburb.nodes[start_id]['x']
            y_start = suburb.nodes[start_id]['y']
            # add the new nodes and geoms in either direction
            for dist in [distance, -distance]:
                # create the end coordinates
                if x_direction:
                    x_centroid = x_start + dist / 2
                    y_centroid = y_start
                    x_end = x_start + dist
                    y_end = y_start
                else:
                    x_centroid = x_start
                    y_centroid = y_start + dist / 2
                    x_end = x_start
                    y_end = y_start + dist
                # calculate the new centroids and end nodes
                centroid_id = node_id
                node_id += 1
                new_centroids.append(centroid_id)  # add to new centroids
                suburb.add_node(centroid_id, x=x_centroid, y=y_centroid)
                end_id = node_id
                node_id += 1
                suburb.add_node(end_id, x=x_end, y=y_end)
                # create the new geoms and edges
                geom_a = geometry.LineString([(x_start, y_start), (x_centroid, y_centroid)])
                suburb.add_edge(start_id, centroid_id, geom=geom_a)
                sum_lengths += geom_a.length
                geom_b = geometry.LineString([(x_centroid, y_centroid), (x_end, y_end)])
                suburb.add_edge(centroid_id, end_id, geom=geom_b)
                sum_lengths += geom_b.length
                # keep track of least length
                last_length = geom_a.length
        centroids = new_centroids

    #  decompose new edges
    suburb = graphs.nX_decompose(suburb, 20)
    # print info
    print(nx.info(suburb))
    # report sum
    print(f'Suburb summed lengths: {sum_lengths}. Last length {last_length}')

    return suburb


# %%
if __name__ == '__main__':
    york_burb_graph = york_burb()
    # plot.plot_nX(york_burb_graph, figsize=(20, 20), dpi=150, labels=False, path='./temp_images/temp_york_burb.png')

    grid_ville_graph = grid_ville()
    # plot.plot_nX(grid_ville_graph, figsize=(20, 20), dpi=150, labels=False, path='./temp_images/temp_grid_ville.png')

    suburb_graph = suburb()
    # plot.plot_nX(suburb_graph, figsize=(20, 20), dpi=150, labels=False, path='./temp_images/temp_suburb.png')
