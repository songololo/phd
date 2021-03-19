import asyncio
import datetime
import logging
import time

import asyncpg
import numpy as np
from cityseer.metrics import layers, networks

from src.process.loaders import postGIS_to_landuses_dict, postGIS_to_networkX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pairwise_disparity_weight(c_i, c_j, weight_1=1 / 3, weight_2=2 / 3, weight_3=3 / 3):
    '''
    Note that this is based on OS POI data, and the disparity methods work accordingly

    Weight can be topological, e.g. 1, 2, 3 or 1, 2, 5 or 0.25, 0.5, 1
    OR weighted per Warwick and Clarke based on decrease in taxonomic diversity
    Warwick and Clarke taxonomic distance - Clarke & Warwick: Weighting step lengths for taxonomic distinctness
    note that this doesn't necessarily add much benefit to topological distance
    also that it constrains the measure to the specific index

    in the case of OS - POI, there are 9 categories with 52 subcategories and 618 unique types
    therefore (see diversity_weights.py)
    in [618, 52, 9]
    out steps [0.916, 0.827, 0.889]
    additive steps [0.916, 1.743, 2.632]
    scaled [34.801, 66.223, 100.0]
    i.e. 1, 2, 3
    '''
    # cast to int
    c_i = int(c_i)
    c_j = int(c_j)
    # calculate 3rd level disparity
    a_rem = c_i % 1000000
    b_rem = c_j % 1000000
    # if the classes diverge at the highest level, return the heaviest weight
    if c_i - a_rem != c_j - b_rem:
        return weight_3
    else:
        # else calculate 2nd level disparity, etc.
        a_small_rem = c_i % 10000
        b_small_rem = c_j % 10000
        if a_rem - a_small_rem != b_rem - b_small_rem:
            return weight_2
        # else default used - which means the disparity is at level 1
        else:
            return weight_1


def disparity_wt_matrix(landuse_classes):
    # prepare the weights matrix
    wt_matrix = np.full((len(landuse_classes), len(landuse_classes)), np.inf)
    for i_idx in range(len(landuse_classes)):
        for j_idx in range(len(landuse_classes)):
            if i_idx == j_idx:
                wt_matrix[i_idx][j_idx] = 0
            else:
                w = pairwise_disparity_weight(landuse_classes[i_idx], landuse_classes[j_idx])
                wt_matrix[i_idx][j_idx] = w
                wt_matrix[j_idx][i_idx] = w
    return wt_matrix


class Accessibility_Codes():

    def __init__(self, landuse_classes, n_elements, distances, compact=False):

        self._class_labels = landuse_classes
        self._n_elements = n_elements
        self._distances = distances

        # generate the class categories by filtering ranges per OS POI categories
        if not compact:
            self.class_categories = {
                'accommodation': [cl for cl in landuse_classes if int(cl) >= 1010000 and int(cl) < 1020000],
                'eating': [cl for cl in landuse_classes if
                           (int(cl) >= 1020000 and int(cl) < 1020034) or
                           (int(cl) >= 1020043 and int(cl) < 2000000)],
                'drinking': [cl for cl in landuse_classes if int(cl) >= 1020034 and int(cl) < 1020043],
                'commercial': [cl for cl in landuse_classes if int(cl) >= 2000000 and int(cl) < 3000000],
                'tourism': [cl for cl in landuse_classes if int(cl) >= 3200000 and int(cl) < 3580000],
                'entertainment': [cl for cl in landuse_classes if int(cl) >= 4250000 and int(cl) < 5000000],
                'government': [cl for cl in landuse_classes if int(cl) >= 6000000 and int(cl) < 7000000],
                'manufacturing': [cl for cl in landuse_classes if int(cl) >= 7000000 and int(cl) < 8000000],
                'retail_food': [cl for cl in landuse_classes if int(cl) >= 9470000 and int(cl) < 9480000],
                'retail_other': [cl for cl in landuse_classes if
                                 (int(cl) >= 9000000 and int(cl) < 9470000) or
                                 (int(cl) >= 9480000 and int(cl) < 10000000)],
                'transport': [cl for cl in landuse_classes if int(cl) >= 10000000 and int(cl) < 11000000],
                'health': [cl for cl in landuse_classes if int(cl) >= 5280000 and int(cl) < 5310000],
                'education': [cl for cl in landuse_classes if int(cl) >= 5310000 and int(cl) < 6000000],
                'parks': [cl for cl in landuse_classes if int(cl) >= 3180000 and int(cl) < 3190000],
                'cultural': [cl for cl in landuse_classes if int(cl) >= 3170000 and int(cl) < 3180000],
                'sports': [cl for cl in landuse_classes if int(cl) >= 4000000 and int(cl) < 4250000]
            }
        else:
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
        self.metrics = {}
        for theme in ['weighted', 'non_weighted']:
            self.metrics[theme] = {}
            for cat in self.class_categories.keys():
                self.metrics[theme][cat] = {}
                for dist in distances:
                    self.metrics[theme][cat][dist] = np.full(n_elements, 0.0)

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
        for theme in self.metrics.keys():
            for cat in self.metrics[theme].keys():
                for dist in self.metrics[theme][cat].keys():
                    for class_code in self.class_categories[cat]:
                        # array-wise
                        self.metrics[theme][cat][dist] += access_metrics_dict[theme][class_code][dist]
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
                             city_pop_id,
                             distances,
                             boundary_table='analysis.city_boundaries_150',
                             data_table='os.poi',
                             data_where=None,
                             rdm_flag=False,
                             dual_flag=False):
    if dual_flag or rdm_flag:
        if city_pop_id > 1:
            logger.warning('Only do dual or randomised metrics for city_pop_id = 1')
            return

    if dual_flag:
        nodes_table += '_dual'
        links_table += '_dual'

    if rdm_flag:
        data_table += '_randomised'

    logger.info(
        f'Starting LU calcs for city id: {city_pop_id} on network table {nodes_table} and data table {data_table}')
    logger.info(f'Loading network data')
    G = await postGIS_to_networkX(db_config, nodes_table, links_table, city_pop_id)
    N = networks.NetworkLayerFromNX(G, distances)
    logger.info(f'Loading POI data from data table: {data_table}')
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
    logger.info('Creating data layer')
    D = layers.DataLayer(data_uids, data_map)

    start = time.localtime()
    logger.info('Assigning data points to the network')
    D.assign_to_network(N, max_dist=400)

    # generate the accessibility codes Class
    # this deduces codes and squashes results into categories
    logger.info('Generating POI accessibility codes')
    Acc_codes = Accessibility_Codes(landuse_classes,
                                    len(N.uids),
                                    distances,
                                    compact=(dual_flag or rdm_flag))

    mixed_use_metrics = ['hill',
                         'hill_branch_wt',
                         'hill_pairwise_wt',
                         'hill_pairwise_disparity',
                         'shannon',
                         'gini_simpson',
                         'raos_pairwise_disparity']
    # if dual or rdm only do first two
    if dual_flag or rdm_flag:
        mixed_use_metrics = mixed_use_metrics[:2]
        cl_disparity_wt_matrix = None
    # compute
    logger.info('Computing landuses')
    D.compute_aggregated(landuse_labels=landuse_labels,
                         mixed_use_keys=mixed_use_metrics,
                         accessibility_keys=Acc_codes.all_codes,
                         cl_disparity_wt_matrix=cl_disparity_wt_matrix,
                         qs=[0, 1, 2])
    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

    # squash the accessibility data
    logger.info('Squashing accessibility data')
    Acc_codes.set_metrics(N.metrics['accessibility'])

    mu_q_keys = ['hill',
                 'hill_branch_wt',
                 'hill_pairwise_wt',
                 'hill_pairwise_disparity']
    if dual_flag or rdm_flag:
        mu_q_keys = mu_q_keys[:2]

    mu_keys = ['shannon',
               'gini_simpson',
               'raos_pairwise_disparity']
    if dual_flag or rdm_flag:
        mu_keys = []

    if not dual_flag and not rdm_flag:
        ac_keys = ['accommodation',
                   'eating',
                   'drinking',
                   'commercial',
                   'tourism',
                   'entertainment',
                   'government',
                   'manufacturing',
                   'retail_food',
                   'retail_other',
                   'transport',
                   'health',
                   'education',
                   'parks',
                   'cultural',
                   'sports',
                   'total']
    else:
        ac_keys = [
            'eating',
            'drinking',
            'commercial',
            'retail_food',
            'retail_other',
            'transport',
            'total'
        ]

    # aggregate the data
    logger.info('Aggregating results')
    bulk_data = []
    for idx, uid in enumerate(N.uids):
        # first check that this is a live node (i.e. within the original city boundary)
        if not N.live[idx]:
            continue
        node_data = [uid]
        # mixed-use keys requiring q values
        for mu_key in mu_q_keys:
            for q_key, q_val in N.metrics['mixed_uses'][mu_key].items():
                inner_data = []
                for d_key, d_val in q_val.items():
                    inner_data.append(d_val[idx])
                node_data.append(inner_data)
        # mixed-use keys not requiring q values
        for mu_key in mu_keys:
            inner_data = []
            for d_key, d_val in N.metrics['mixed_uses'][mu_key].items():
                inner_data.append(d_val[idx])
            node_data.append(inner_data)
        # accessibility keys
        for ac_key in ac_keys:
            inner_data = []
            for d_key, d_val in Acc_codes.metrics['weighted'][ac_key].items():
                inner_data.append(d_val[idx])
            node_data.append(inner_data)
            # also write non-weighted variants of the following
            if ac_key in ['eating', 'commercial', 'retail_food', 'retail_other', 'total']:
                inner_data = []
                for d_key, d_val in Acc_codes.metrics['non_weighted'][ac_key].items():
                    inner_data.append(d_val[idx])
                node_data.append(inner_data)
        bulk_data.append(tuple(node_data))

    logger.info('Writing results to database')
    db_con = await asyncpg.connect(**db_config)
    if not dual_flag and not rdm_flag:
        measure_cols = ['mu_hill_0',
                        'mu_hill_1',
                        'mu_hill_2',
                        'mu_hill_branch_wt_0',
                        'mu_hill_branch_wt_1',
                        'mu_hill_branch_wt_2',
                        'mu_hill_pairwise_wt_0',
                        'mu_hill_pairwise_wt_1',
                        'mu_hill_pairwise_wt_2',
                        'mu_hill_dispar_wt_0',
                        'mu_hill_dispar_wt_1',
                        'mu_hill_dispar_wt_2',
                        'mu_shannon',
                        'mu_gini',
                        'mu_raos',
                        'ac_accommodation',
                        'ac_eating',
                        'ac_eating_nw',
                        'ac_drinking',
                        'ac_commercial',
                        'ac_commercial_nw',
                        'ac_tourism',
                        'ac_entertainment',
                        'ac_government',
                        'ac_manufacturing',
                        'ac_retail_food',
                        'ac_retail_food_nw',
                        'ac_retail_other',
                        'ac_retail_other_nw',
                        'ac_transport',
                        'ac_health',
                        'ac_education',
                        'ac_parks',
                        'ac_cultural',
                        'ac_sports',
                        'ac_total',
                        'ac_total_nw']
    else:
        measure_cols = [
            'mu_hill_0',
            'mu_hill_1',
            'mu_hill_2',
            'mu_hill_branch_wt_0',
            'mu_hill_branch_wt_1',
            'mu_hill_branch_wt_2',
            'ac_eating',
            'ac_eating_nw',
            'ac_drinking',
            'ac_commercial',
            'ac_commercial_nw',
            'ac_retail_food',
            'ac_retail_food_nw',
            'ac_retail_other',
            'ac_retail_other_nw',
            'ac_transport',
            'ac_total',
            'ac_total_nw'
        ]
    # add the _rdm extension if necessary
    if rdm_flag:
        measure_cols = [m + '_rdm' for m in measure_cols]
    # create the columns
    col_strings = []
    counter = 2
    for measure_col in measure_cols:
        await db_con.execute(f'ALTER TABLE {nodes_table} ADD COLUMN IF NOT EXISTS {measure_col} real[];')
        col_strings.append(f'{measure_col} = ${counter}')
        counter += 1
    await db_con.executemany(f'UPDATE {nodes_table} SET ' + ', '.join(col_strings) + ' WHERE id = $1;', bulk_data)
    await db_con.close()


if __name__ == '__main__':

    ######################################################################################
    # NOTE -> if newer data in table than date last updated then won't find correct data #
    ######################################################################################
    async def fetch_date_last_updated(db_config):
        db_con = await asyncpg.connect(**db_config)
        d_u = await db_con.fetchval(f'''
            select date_last_updated from os.poi order by date_last_updated desc limit 1; 
        ''')
        logger.info(f'Using {d_u} as the DATE_LAST_UPDATED parameter.')
        return d_u


    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }
    date_last_updated = asyncio.run(fetch_date_last_updated(db_config))
    data_where = f"date_last_updated = date('{date_last_updated}')"
    max_dist = 1600
    distances = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    nodes_table = 'analysis.nodes_20'
    links_table = 'analysis.links_20'
    rdm = False
    dual = False

    for city_pop_id in range(362, 932):
        start_time = time.localtime()
        logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')
        asyncio.run(
            accessibility_calc(db_config,
                               nodes_table,
                               links_table,
                               city_pop_id,
                               distances,
                               data_where=data_where,
                               rdm_flag=rdm,
                               dual_flag=dual))
        logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')
        end_time = time.localtime()
        logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
