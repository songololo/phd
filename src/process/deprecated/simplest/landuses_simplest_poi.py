import asyncio
import datetime
import logging
import time

import asyncpg
import numpy as np
from cityseer.metrics import layers, networks
from cityseer.util import graphs
from process.loaders import postGIS_to_landuses_dict, postGIS_to_networkX
from process.metrics.shortest.landuses_shortest_poi import disparity_wt_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Accessibility_Codes():

    def __init__(self, landuse_classes, n_elements, distances):

        self._class_labels = landuse_classes
        self._n_elements = n_elements
        self._distances = distances

        # generate the class categories by filtering ranges per OS POI categories
        self.class_categories = {
            'eating': [cl for cl in landuse_classes if
                       (int(cl) >= 1020000 and int(cl) < 1020034) or
                       (int(cl) >= 1020043 and int(cl) < 2000000)],
            'drinking': [cl for cl in landuse_classes if int(cl) >= 1020034 and int(cl) < 1020043],
            'commercial': [cl for cl in landuse_classes if int(cl) >= 2000000 and int(cl) < 3000000],
            'retail_food': [cl for cl in landuse_classes if int(cl) >= 9470000 and int(cl) < 9480000],
            'retail_other': [cl for cl in landuse_classes if
                             (int(cl) >= 9000000 and int(cl) < 9470000) or
                             (int(cl) >= 9480000 and int(cl) < 10000000)],
            'transport': [cl for cl in landuse_classes if int(cl) >= 10000000 and int(cl) < 11000000]
        }

        # prepare data structure for unpacking results
        self.metrics = {
            'non_weighted': {}
        }
        for cat in self.class_categories.keys():
            self.metrics['non_weighted'][cat] = {}
            for dist in distances:
                self.metrics['non_weighted'][cat][dist] = np.full(n_elements, 0.0)

    @property
    def categories(self):
        return [k for k in self.class_categories.keys()]

    @property
    def all_codes(self):
        class_codes = []
        for v in self.class_categories.values():
            class_codes += v
        return class_codes

    def set_metrics(self, access_metrics_dict):
        for cat in self.metrics['non_weighted'].keys():
            for dist in self.metrics['non_weighted'][cat].keys():
                for class_code in self.class_categories[cat]:
                    self.metrics['non_weighted'][cat][dist] += access_metrics_dict['non_weighted'][class_code][dist]
        # add totals
        # pre-total categories
        for wt in self.metrics.keys():
            self.metrics[wt]['total'] = {}
            for dist in self._distances:
                self.metrics[wt]['total'][dist] = np.full(self._n_elements, 0.0)
                for cat in self.categories:
                    # array-wise
                    self.metrics[wt]['total'][dist] += self.metrics[wt][cat][dist]


async def accessibility_calc(db_config,
                             nodes_table,
                             links_table,
                             boundary_table,
                             data_table,
                             city_pop_id,
                             distances,
                             data_where=None,
                             random=False):
    logger.info('Checking for tables')

    # CREATE METRICS roanodes tables
    db_con = await asyncpg.connect(**db_config)

    # random is for randomised controls / comparison
    if not random:
        rn_lu_table = nodes_table + '_dual_lu'
    else:
        rn_lu_table = nodes_table + '_dual_lu_randomised'
    rn_lu_table_only = rn_lu_table.split('.')[-1]

    rn_lu_exists = await db_con.fetchval(f'''
                        SELECT EXISTS (
                           SELECT 1
                           FROM   information_schema.tables 
                           WHERE  table_schema = 'analysis'
                           AND    table_name = '{rn_lu_table_only}'
                        );
                    ''')

    if rn_lu_exists:
        logger.info(f'Roadnodes landuses table {rn_lu_table} already exists, skipping...')
    else:
        logger.info(f'Copying base columns from roadnodes table: {nodes_table}_dual to landuses table: {rn_lu_table}')
        base_cols = [
            'id',
            'geom',
            'city_pop_id'
        ]
        # don't filter by city_pop_id because still need buffered locations
        await db_con.execute(f'''
                CREATE TABLE {rn_lu_table} AS SELECT {', '.join(base_cols)} FROM {nodes_table}_dual;
                ALTER TABLE {rn_lu_table} ADD PRIMARY KEY (id);
                CREATE INDEX geom_idx_{rn_lu_table_only} ON {rn_lu_table} USING GIST (geom);
                CREATE INDEX pop_idx_{rn_lu_table_only} ON {rn_lu_table} (city_pop_id);
            ''')

    # add the new columns
    logger.info(f'Adding landuses columns, if necessary, to {rn_lu_table}')

    for theme in [
        'mu_hill_0_simpl_{dist}',
        'mu_hill_0_short_{dist}',
        'mu_hill_1_simpl_{dist}',
        'mu_hill_1_short_{dist}',
        'mu_hill_2_simpl_{dist}',
        'mu_hill_2_short_{dist}',
        'mu_hill_branch_wt_0_simpl_{dist}',
        'mu_hill_branch_wt_0_short_{dist}',
        'mu_hill_branch_wt_1_simpl_{dist}',
        'mu_hill_branch_wt_1_short_{dist}',
        'mu_hill_branch_wt_2_simpl_{dist}',
        'mu_hill_branch_wt_2_short_{dist}',
        'mu_hill_pairwise_wt_0_simpl_{dist}',
        'mu_hill_pairwise_wt_0_short_{dist}',
        'mu_hill_pairwise_wt_1_simpl_{dist}',
        'mu_hill_pairwise_wt_1_short_{dist}',
        'mu_hill_pairwise_wt_2_simpl_{dist}',
        'mu_hill_pairwise_wt_2_short_{dist}',
        'ac_eating_nw_simpl_{dist}',
        'ac_eating_nw_short_{dist}',
        'ac_drinking_nw_simpl_{dist}',
        'ac_drinking_nw_short_{dist}',
        'ac_commercial_nw_simpl_{dist}',
        'ac_commercial_nw_short_{dist}',
        'ac_retail_food_nw_simpl_{dist}',
        'ac_retail_food_nw_short_{dist}',
        'ac_retail_other_nw_simpl_{dist}',
        'ac_retail_other_nw_short_{dist}',
        'ac_transport_nw_simpl_{dist}',
        'ac_transport_nw_short_{dist}',
        'ac_total_nw_simpl_{dist}',
        'ac_total_nw_short_{dist}']:
        for d_key in distances:
            await db_con.execute(f'''
                    ALTER TABLE {rn_lu_table} ADD COLUMN IF NOT EXISTS {theme.format(dist=d_key)} real;
                ''')

    await db_con.close()

    logger.info(
        f'Starting simplest-path landuse calcs for city id: {city_pop_id} on network table {nodes_table} and data table {data_table}')

    logger.info(f'Loading network data')
    G = await postGIS_to_networkX(db_config, nodes_table, links_table, boundary_table, city_pop_id)
    G_dual = graphs.nX_to_dual(G)  # convert to dual

    start = time.localtime()

    logger.info(f'Loading POI data')
    data_dict = await postGIS_to_landuses_dict(db_config,
                                               data_table,
                                               'urn',
                                               'class_code',
                                               boundary_table,
                                               city_pop_id,
                                               max_dist=max(distances),
                                               data_where=data_where)
    data_uids, data_map = layers.data_map_from_dict(data_dict)
    # derive the landuse labels, classes, encodings
    landuse_labels = [v['class'] for v in data_dict.values()]
    landuse_classes, landuse_encodings = layers.encode_categorical(landuse_labels)

    logger.info(f'Generating disparity weights matrix')
    cl_disparity_wt_matrix = disparity_wt_matrix(landuse_classes)

    # SIMPLEST PATH
    logger.info(f'Generating simplest paths network and data layers')
    N_simplest = networks.Network_Layer_From_nX(G_dual,
                                                distances=distances,
                                                angular=True)
    D_simplest = layers.Data_Layer(data_uids, data_map)
    logger.info('Assigning data points to the network (simplest)')
    D_simplest.assign_to_network(N_simplest, max_dist=400)
    # generate the accessibility codes Class - this deduces codes and squashes results into categories
    logger.info('Generating POI accessibility codes')
    Acc_codes_simplest = Accessibility_Codes(landuse_classes, len(N_simplest.uids), distances)
    logger.info('Computing mixed-uses and accessibility (simplest)')
    D_simplest.compute_aggregated(landuse_labels=landuse_labels,
                                  mixed_use_keys=['hill', 'hill_branch_wt', 'hill_pairwise_wt'],
                                  accessibility_keys=Acc_codes_simplest.all_codes,
                                  cl_disparity_wt_matrix=cl_disparity_wt_matrix,
                                  qs=[0, 1, 2])
    # squash the accessibility data
    logger.info('Squashing accessibility data')
    Acc_codes_simplest.set_metrics(N_simplest.metrics['accessibility'])

    # SHORTEST PATH
    logger.info(f'Generating centrality paths network and data layers')
    N_shortest = networks.Network_Layer_From_nX(G_dual,
                                                distances=distances,
                                                angular=False)
    # override angular distances with normal distances
    assert not np.array_equal(N_shortest.edge_impedances, N_shortest.edge_lengths)
    # impedances are stored in index 3
    N_shortest._edges[:, 3] = N_shortest.edge_lengths
    assert np.array_equal(N_shortest.edge_impedances, N_shortest.edge_lengths)
    # create the data layer
    D_shortest = layers.Data_Layer(data_uids, data_map)
    logger.info('Assigning data points to the network (centrality)')
    # TODO: Optional... could save some time here by copying network assignment from simplest?
    D_shortest.assign_to_network(N_shortest, max_dist=400)
    # generate the accessibility codes Class - this deduces codes and squashes results into categories
    logger.info('Generating POI accessibility codes')
    Acc_codes_shortest = Accessibility_Codes(landuse_classes, len(N_shortest.uids), distances)
    logger.info('Computing mixed-uses and accessibility (centrality)')
    D_shortest.compute_aggregated(landuse_labels=landuse_labels,
                                  mixed_use_keys=['hill', 'hill_branch_wt', 'hill_pairwise_wt'],
                                  accessibility_keys=Acc_codes_shortest.all_codes,
                                  cl_disparity_wt_matrix=cl_disparity_wt_matrix,
                                  qs=[0, 1, 2])
    # squash the accessibility data
    logger.info('Squashing accessibility data')
    Acc_codes_shortest.set_metrics(N_shortest.metrics['accessibility'])

    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

    # aggregate the data
    logger.info('Aggregating results')
    bulk_data = []

    for idx, (uid_simplest, uid_shortest) in enumerate(zip(N_simplest.uids, N_shortest.uids)):
        assert uid_simplest == uid_shortest

        # first check that this is a live node (i.e. within the original city boundary)
        if not N_simplest.live[idx]:
            continue

        node_data = [uid_simplest]

        for mu_key in ['hill',
                       'hill_branch_wt',
                       'hill_pairwise_wt']:
            for q_key in [0, 1, 2]:
                # simplest path first
                for d_val_simplest in N_simplest.metrics['mixed_uses'][mu_key][q_key].values():
                    node_data.append(d_val_simplest[idx])
                # then centrality path
                for d_val_shortest in N_shortest.metrics['mixed_uses'][mu_key][q_key].values():
                    node_data.append(d_val_shortest[idx])

        for ac_key in ['eating',
                       'drinking',
                       'commercial',
                       'retail_food',
                       'retail_other',
                       'transport',
                       'total']:
            # simplest path first
            for d_val_simplest in Acc_codes_simplest.metrics['non_weighted'][ac_key].values():
                node_data.append(d_val_simplest[idx])
            # then centrality path
            for d_val_shortest in Acc_codes_shortest.metrics['non_weighted'][ac_key].values():
                node_data.append(d_val_shortest[idx])

        bulk_data.append(tuple(node_data))

    logger.info('Writing results to database')
    db_con = await asyncpg.connect(**db_config)
    await db_con.executemany(f'''
        UPDATE {rn_lu_table}
            SET
                mu_hill_0_simpl_50 = $2,
                mu_hill_0_simpl_100 = $3,
                mu_hill_0_simpl_150 = $4,
                mu_hill_0_simpl_200 = $5,
                mu_hill_0_simpl_300 = $6,
                mu_hill_0_simpl_400 = $7,
                mu_hill_0_simpl_600 = $8,
                mu_hill_0_simpl_800 = $9,
                mu_hill_0_simpl_1200 = $10,
                mu_hill_0_simpl_1600 = $11,
                mu_hill_0_short_50 = $12,
                mu_hill_0_short_100 = $13,
                mu_hill_0_short_150 = $14,
                mu_hill_0_short_200 = $15,
                mu_hill_0_short_300 = $16,
                mu_hill_0_short_400 = $17,
                mu_hill_0_short_600 = $18,
                mu_hill_0_short_800 = $19,
                mu_hill_0_short_1200 = $20,
                mu_hill_0_short_1600 = $21,
                mu_hill_1_simpl_50 = $22,
                mu_hill_1_simpl_100 = $23,
                mu_hill_1_simpl_150 = $24,
                mu_hill_1_simpl_200 = $25,
                mu_hill_1_simpl_300 = $26,
                mu_hill_1_simpl_400 = $27,
                mu_hill_1_simpl_600 = $28,
                mu_hill_1_simpl_800 = $29,
                mu_hill_1_simpl_1200 = $30,
                mu_hill_1_simpl_1600 = $31,
                mu_hill_1_short_50 = $32,
                mu_hill_1_short_100 = $33,
                mu_hill_1_short_150 = $34,
                mu_hill_1_short_200 = $35,
                mu_hill_1_short_300 = $36,
                mu_hill_1_short_400 = $37,
                mu_hill_1_short_600 = $38,
                mu_hill_1_short_800 = $39,
                mu_hill_1_short_1200 = $40,
                mu_hill_1_short_1600 = $41,
                mu_hill_2_simpl_50 = $42,
                mu_hill_2_simpl_100 = $43,
                mu_hill_2_simpl_150 = $44,
                mu_hill_2_simpl_200 = $45,
                mu_hill_2_simpl_300 = $46,
                mu_hill_2_simpl_400 = $47,
                mu_hill_2_simpl_600 = $48,
                mu_hill_2_simpl_800 = $49,
                mu_hill_2_simpl_1200 = $50,
                mu_hill_2_simpl_1600 = $51,
                mu_hill_2_short_50 = $52,
                mu_hill_2_short_100 = $53,
                mu_hill_2_short_150 = $54,
                mu_hill_2_short_200 = $55,
                mu_hill_2_short_300 = $56,
                mu_hill_2_short_400 = $57,
                mu_hill_2_short_600 = $58,
                mu_hill_2_short_800 = $59,
                mu_hill_2_short_1200 = $60,
                mu_hill_2_short_1600 = $61,
                mu_hill_branch_wt_0_simpl_50 = $62,
                mu_hill_branch_wt_0_simpl_100 = $63,
                mu_hill_branch_wt_0_simpl_150 = $64,
                mu_hill_branch_wt_0_simpl_200 = $65,
                mu_hill_branch_wt_0_simpl_300 = $66,
                mu_hill_branch_wt_0_simpl_400 = $67,
                mu_hill_branch_wt_0_simpl_600 = $68,
                mu_hill_branch_wt_0_simpl_800 = $69,
                mu_hill_branch_wt_0_simpl_1200 = $70,
                mu_hill_branch_wt_0_simpl_1600 = $71,
                mu_hill_branch_wt_0_short_50 = $72,
                mu_hill_branch_wt_0_short_100 = $73,
                mu_hill_branch_wt_0_short_150 = $74,
                mu_hill_branch_wt_0_short_200 = $75,
                mu_hill_branch_wt_0_short_300 = $76,
                mu_hill_branch_wt_0_short_400 = $77,
                mu_hill_branch_wt_0_short_600 = $78,
                mu_hill_branch_wt_0_short_800 = $79,
                mu_hill_branch_wt_0_short_1200 = $80,
                mu_hill_branch_wt_0_short_1600 = $81,
                mu_hill_branch_wt_1_simpl_50 = $82,
                mu_hill_branch_wt_1_simpl_100 = $83,
                mu_hill_branch_wt_1_simpl_150 = $84,
                mu_hill_branch_wt_1_simpl_200 = $85,
                mu_hill_branch_wt_1_simpl_300 = $86,
                mu_hill_branch_wt_1_simpl_400 = $87,
                mu_hill_branch_wt_1_simpl_600 = $88,
                mu_hill_branch_wt_1_simpl_800 = $89,
                mu_hill_branch_wt_1_simpl_1200 = $90,
                mu_hill_branch_wt_1_simpl_1600 = $91,
                mu_hill_branch_wt_1_short_50 = $92,
                mu_hill_branch_wt_1_short_100 = $93,
                mu_hill_branch_wt_1_short_150 = $94,
                mu_hill_branch_wt_1_short_200 = $95,
                mu_hill_branch_wt_1_short_300 = $96,
                mu_hill_branch_wt_1_short_400 = $97,
                mu_hill_branch_wt_1_short_600 = $98,
                mu_hill_branch_wt_1_short_800 = $99,
                mu_hill_branch_wt_1_short_1200 = $100,
                mu_hill_branch_wt_1_short_1600 = $101,
                mu_hill_branch_wt_2_simpl_50 = $102,
                mu_hill_branch_wt_2_simpl_100 = $103,
                mu_hill_branch_wt_2_simpl_150 = $104,
                mu_hill_branch_wt_2_simpl_200 = $105,
                mu_hill_branch_wt_2_simpl_300 = $106,
                mu_hill_branch_wt_2_simpl_400 = $107,
                mu_hill_branch_wt_2_simpl_600 = $108,
                mu_hill_branch_wt_2_simpl_800 = $109,
                mu_hill_branch_wt_2_simpl_1200 = $110,
                mu_hill_branch_wt_2_simpl_1600 = $111,
                mu_hill_branch_wt_2_short_50 = $112,
                mu_hill_branch_wt_2_short_100 = $113,
                mu_hill_branch_wt_2_short_150 = $114,
                mu_hill_branch_wt_2_short_200 = $115,
                mu_hill_branch_wt_2_short_300 = $116,
                mu_hill_branch_wt_2_short_400 = $117,
                mu_hill_branch_wt_2_short_600 = $118,
                mu_hill_branch_wt_2_short_800 = $119,
                mu_hill_branch_wt_2_short_1200 = $120,
                mu_hill_branch_wt_2_short_1600 = $121,
                mu_hill_pairwise_wt_0_simpl_50 = $122,
                mu_hill_pairwise_wt_0_simpl_100 = $123,
                mu_hill_pairwise_wt_0_simpl_150 = $124,
                mu_hill_pairwise_wt_0_simpl_200 = $125,
                mu_hill_pairwise_wt_0_simpl_300 = $126,
                mu_hill_pairwise_wt_0_simpl_400 = $127,
                mu_hill_pairwise_wt_0_simpl_600 = $128,
                mu_hill_pairwise_wt_0_simpl_800 = $129,
                mu_hill_pairwise_wt_0_simpl_1200 = $130,
                mu_hill_pairwise_wt_0_simpl_1600 = $131,
                mu_hill_pairwise_wt_0_short_50 = $132,
                mu_hill_pairwise_wt_0_short_100 = $133,
                mu_hill_pairwise_wt_0_short_150 = $134,
                mu_hill_pairwise_wt_0_short_200 = $135,
                mu_hill_pairwise_wt_0_short_300 = $136,
                mu_hill_pairwise_wt_0_short_400 = $137,
                mu_hill_pairwise_wt_0_short_600 = $138,
                mu_hill_pairwise_wt_0_short_800 = $139,
                mu_hill_pairwise_wt_0_short_1200 = $140,
                mu_hill_pairwise_wt_0_short_1600 = $141,
                mu_hill_pairwise_wt_1_simpl_50 = $142,
                mu_hill_pairwise_wt_1_simpl_100 = $143,
                mu_hill_pairwise_wt_1_simpl_150 = $144,
                mu_hill_pairwise_wt_1_simpl_200 = $145,
                mu_hill_pairwise_wt_1_simpl_300 = $146,
                mu_hill_pairwise_wt_1_simpl_400 = $147,
                mu_hill_pairwise_wt_1_simpl_600 = $148,
                mu_hill_pairwise_wt_1_simpl_800 = $149,
                mu_hill_pairwise_wt_1_simpl_1200 = $150,
                mu_hill_pairwise_wt_1_simpl_1600 = $151,
                mu_hill_pairwise_wt_1_short_50 = $152,
                mu_hill_pairwise_wt_1_short_100 = $153,
                mu_hill_pairwise_wt_1_short_150 = $154,
                mu_hill_pairwise_wt_1_short_200 = $155,
                mu_hill_pairwise_wt_1_short_300 = $156,
                mu_hill_pairwise_wt_1_short_400 = $157,
                mu_hill_pairwise_wt_1_short_600 = $158,
                mu_hill_pairwise_wt_1_short_800 = $159,
                mu_hill_pairwise_wt_1_short_1200 = $160,
                mu_hill_pairwise_wt_1_short_1600 = $161,
                mu_hill_pairwise_wt_2_simpl_50 = $162,
                mu_hill_pairwise_wt_2_simpl_100 = $163,
                mu_hill_pairwise_wt_2_simpl_150 = $164,
                mu_hill_pairwise_wt_2_simpl_200 = $165,
                mu_hill_pairwise_wt_2_simpl_300 = $166,
                mu_hill_pairwise_wt_2_simpl_400 = $167,
                mu_hill_pairwise_wt_2_simpl_600 = $168,
                mu_hill_pairwise_wt_2_simpl_800 = $169,
                mu_hill_pairwise_wt_2_simpl_1200 = $170,
                mu_hill_pairwise_wt_2_simpl_1600 = $171,
                mu_hill_pairwise_wt_2_short_50 = $172,
                mu_hill_pairwise_wt_2_short_100 = $173,
                mu_hill_pairwise_wt_2_short_150 = $174,
                mu_hill_pairwise_wt_2_short_200 = $175,
                mu_hill_pairwise_wt_2_short_300 = $176,
                mu_hill_pairwise_wt_2_short_400 = $177,
                mu_hill_pairwise_wt_2_short_600 = $178,
                mu_hill_pairwise_wt_2_short_800 = $179,
                mu_hill_pairwise_wt_2_short_1200 = $180,
                mu_hill_pairwise_wt_2_short_1600 = $181,
                ac_eating_nw_simpl_50 = $182,
                ac_eating_nw_simpl_100 = $183,
                ac_eating_nw_simpl_150 = $184,
                ac_eating_nw_simpl_200 = $185,
                ac_eating_nw_simpl_300 = $186,
                ac_eating_nw_simpl_400 = $187,
                ac_eating_nw_simpl_600 = $188,
                ac_eating_nw_simpl_800 = $189,
                ac_eating_nw_simpl_1200 = $190,
                ac_eating_nw_simpl_1600 = $191,
                ac_eating_nw_short_50 = $192,
                ac_eating_nw_short_100 = $193,
                ac_eating_nw_short_150 = $194,
                ac_eating_nw_short_200 = $195,
                ac_eating_nw_short_300 = $196,
                ac_eating_nw_short_400 = $197,
                ac_eating_nw_short_600 = $198,
                ac_eating_nw_short_800 = $199,
                ac_eating_nw_short_1200 = $200,
                ac_eating_nw_short_1600 = $201,
                ac_drinking_nw_simpl_50 = $202,
                ac_drinking_nw_simpl_100 = $203,
                ac_drinking_nw_simpl_150 = $204,
                ac_drinking_nw_simpl_200 = $205,
                ac_drinking_nw_simpl_300 = $206,
                ac_drinking_nw_simpl_400 = $207,
                ac_drinking_nw_simpl_600 = $208,
                ac_drinking_nw_simpl_800 = $209,
                ac_drinking_nw_simpl_1200 = $210,
                ac_drinking_nw_simpl_1600 = $211,
                ac_drinking_nw_short_50 = $212,
                ac_drinking_nw_short_100 = $213,
                ac_drinking_nw_short_150 = $214,
                ac_drinking_nw_short_200 = $215,
                ac_drinking_nw_short_300 = $216,
                ac_drinking_nw_short_400 = $217,
                ac_drinking_nw_short_600 = $218,
                ac_drinking_nw_short_800 = $219,
                ac_drinking_nw_short_1200 = $220,
                ac_drinking_nw_short_1600 = $221,
                ac_commercial_nw_simpl_50 = $222,
                ac_commercial_nw_simpl_100 = $223,
                ac_commercial_nw_simpl_150 = $224,
                ac_commercial_nw_simpl_200 = $225,
                ac_commercial_nw_simpl_300 = $226,
                ac_commercial_nw_simpl_400 = $227,
                ac_commercial_nw_simpl_600 = $228,
                ac_commercial_nw_simpl_800 = $229,
                ac_commercial_nw_simpl_1200 = $230,
                ac_commercial_nw_simpl_1600 = $231,
                ac_commercial_nw_short_50 = $232,
                ac_commercial_nw_short_100 = $233,
                ac_commercial_nw_short_150 = $234,
                ac_commercial_nw_short_200 = $235,
                ac_commercial_nw_short_300 = $236,
                ac_commercial_nw_short_400 = $237,
                ac_commercial_nw_short_600 = $238,
                ac_commercial_nw_short_800 = $239,
                ac_commercial_nw_short_1200 = $240,
                ac_commercial_nw_short_1600 = $241,
                ac_retail_food_nw_simpl_50 = $242,
                ac_retail_food_nw_simpl_100 = $243,
                ac_retail_food_nw_simpl_150 = $244,
                ac_retail_food_nw_simpl_200 = $245,
                ac_retail_food_nw_simpl_300 = $246,
                ac_retail_food_nw_simpl_400 = $247,
                ac_retail_food_nw_simpl_600 = $248,
                ac_retail_food_nw_simpl_800 = $249,
                ac_retail_food_nw_simpl_1200 = $250,
                ac_retail_food_nw_simpl_1600 = $251,
                ac_retail_food_nw_short_50 = $252,
                ac_retail_food_nw_short_100 = $253,
                ac_retail_food_nw_short_150 = $254,
                ac_retail_food_nw_short_200 = $255,
                ac_retail_food_nw_short_300 = $256,
                ac_retail_food_nw_short_400 = $257,
                ac_retail_food_nw_short_600 = $258,
                ac_retail_food_nw_short_800 = $259,
                ac_retail_food_nw_short_1200 = $260,
                ac_retail_food_nw_short_1600 = $261,
                ac_retail_other_nw_simpl_50 = $262,
                ac_retail_other_nw_simpl_100 = $263,
                ac_retail_other_nw_simpl_150 = $264,
                ac_retail_other_nw_simpl_200 = $265,
                ac_retail_other_nw_simpl_300 = $266,
                ac_retail_other_nw_simpl_400 = $267,
                ac_retail_other_nw_simpl_600 = $268,
                ac_retail_other_nw_simpl_800 = $269,
                ac_retail_other_nw_simpl_1200 = $270,
                ac_retail_other_nw_simpl_1600 = $271,
                ac_retail_other_nw_short_50 = $272,
                ac_retail_other_nw_short_100 = $273,
                ac_retail_other_nw_short_150 = $274,
                ac_retail_other_nw_short_200 = $275,
                ac_retail_other_nw_short_300 = $276,
                ac_retail_other_nw_short_400 = $277,
                ac_retail_other_nw_short_600 = $278,
                ac_retail_other_nw_short_800 = $279,
                ac_retail_other_nw_short_1200 = $280,
                ac_retail_other_nw_short_1600 = $281,
                ac_transport_nw_simpl_50 = $282,
                ac_transport_nw_simpl_100 = $283,
                ac_transport_nw_simpl_150 = $284,
                ac_transport_nw_simpl_200 = $285,
                ac_transport_nw_simpl_300 = $286,
                ac_transport_nw_simpl_400 = $287,
                ac_transport_nw_simpl_600 = $288,
                ac_transport_nw_simpl_800 = $289,
                ac_transport_nw_simpl_1200 = $290,
                ac_transport_nw_simpl_1600 = $291,
                ac_transport_nw_short_50 = $292,
                ac_transport_nw_short_100 = $293,
                ac_transport_nw_short_150 = $294,
                ac_transport_nw_short_200 = $295,
                ac_transport_nw_short_300 = $296,
                ac_transport_nw_short_400 = $297,
                ac_transport_nw_short_600 = $298,
                ac_transport_nw_short_800 = $299,
                ac_transport_nw_short_1200 = $300,
                ac_transport_nw_short_1600 = $301,
                ac_total_nw_simpl_50 = $302,
                ac_total_nw_simpl_100 = $303,
                ac_total_nw_simpl_150 = $304,
                ac_total_nw_simpl_200 = $305,
                ac_total_nw_simpl_300 = $306,
                ac_total_nw_simpl_400 = $307,
                ac_total_nw_simpl_600 = $308,
                ac_total_nw_simpl_800 = $309,
                ac_total_nw_simpl_1200 = $310,
                ac_total_nw_simpl_1600 = $311,
                ac_total_nw_short_50 = $312,
                ac_total_nw_short_100 = $313,
                ac_total_nw_short_150 = $314,
                ac_total_nw_short_200 = $315,
                ac_total_nw_short_300 = $316,
                ac_total_nw_short_400 = $317,
                ac_total_nw_short_600 = $318,
                ac_total_nw_short_800 = $319,
                ac_total_nw_short_1200 = $320,
                ac_total_nw_short_1600 = $321
            WHERE id = $1;
        ''', bulk_data)
    await db_con.close()


if __name__ == '__main__':

    from process.metrics.shortest.landuses_shortest_poi import fetch_date_last_updated

    logging.getLogger('asyncio').setLevel(logging.WARNING)
    loop = asyncio.get_event_loop()
    # loop.set_debug(True)

    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }

    boundary_table = 'analysis.city_boundaries_150'
    data_table = 'os.poi'
    date_last_updated = loop.run_until_complete(fetch_date_last_updated(db_config))
    data_where = f"date_last_updated = date('{date_last_updated}')"

    nodes_table = 'analysis.roadnodes_20'
    links_table = 'analysis.roadlinks_20'

    distances = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]

    for city_pop_id in range(1, 2):
        start_time = time.localtime()
        logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

        loop.run_until_complete(
            accessibility_calc(db_config,
                               nodes_table,
                               links_table,
                               boundary_table,
                               data_table,
                               city_pop_id,
                               distances,
                               data_where=data_where))

        logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

        end_time = time.localtime()
        logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
