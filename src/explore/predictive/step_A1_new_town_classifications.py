# %%
'''
STEP 1:
Uses an extra trees classifier to iteratively label / visually verify / update
new town vs. old town designations for mid-sized towns in Great Britain
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from src import util_funcs
from src.explore.theme_setup import data_path
from src.explore.theme_setup import generate_theme

#  %% load from disk
df_full = pd.read_feather(data_path / 'df_full_all.feather')
df_full = df_full.set_index('id')
X_raw, distances, labels = generate_theme(df_full,
                                          'all_towns',
                                          bandwise=False,
                                          city_pop_id=True)

# db connection params
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}

# load boundaries data
bound_data = util_funcs.load_data_as_pd_df(db_config,
                                           ['pop_id',
                                          'city_name',
                                          'city_type',
                                          'city_area',
                                          'city_area_petite',
                                          'city_population',
                                          'city_species_count',
                                          'city_species_unique',
                                          'city_streets_len',
                                          'city_intersections_count'],
                                           'analysis.city_boundaries_150',
                                           'WHERE pop_id IS NOT NULL ORDER BY pop_id')
# add indices for city-wide data
bound_data.set_index('pop_id', inplace=True)
bound_data.sort_index(inplace=True)

town_classifications = ['New Town',
                        'New Town Like',
                        'Expanded Town',
                        'Expanded Town Like']

selected_data = bound_data.copy()
new_towns = selected_data[
    selected_data.city_type.isin(town_classifications)]
largest = new_towns.city_population.max()
smallest = new_towns.city_population.min()
selected_data = selected_data[np.logical_and(selected_data.city_population >= smallest,
                                             selected_data.city_population <= largest)]
X = pd.DataFrame()
y = pd.DataFrame()
for idx, city in selected_data.iterrows():
    X = X.append(pd.Series(name=idx))
    y = y.append(pd.Series(name=idx))

selected_columns = []
selected_columns_dist = [
    'cens_tot_pop_{dist}',
    'cens_dwellings_{dist}',
    'c_node_harmonic_angular_{dist}',
    'c_node_betweenness_beta_{dist}',
    'mu_hill_branch_wt_0_{dist}',
    'ac_eating_{dist}',
    'ac_drinking_{dist}',
    'ac_commercial_{dist}',
    'ac_manufacturing_{dist}',
    'ac_retail_food_{dist}',
    'ac_retail_other_{dist}',
    'ac_transport_{dist}',
    'ac_total_{dist}']
for pop_id, city in selected_data.iterrows():
    if (city.city_type in town_classifications):
        y.loc[pop_id, 'targets'] = 1
    else:
        y.loc[pop_id, 'targets'] = 0
    for column in selected_columns_dist:  # X_raw.columns:
        for dist in distances:  # only if not using columns directly
            if column == 'city_pop_id':
                continue
            key = column.format(dist=dist)
            selected_columns.append(f'{key}_mean')
            selected_columns.append(f'{key}_std')
            city_data = X_raw.loc[X_raw.city_pop_id == pop_id, key]
            X.loc[pop_id, f'{key}_mean'] = np.nanmean(city_data)
            X.loc[pop_id, f'{key}_std'] = np.nanstd(city_data)
y.targets = y.targets.astype(int)
X_data = X[selected_columns]

X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    y.targets,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    shuffle=True,
                                                    stratify=y.targets)

# %%
'''
GRID SEARCH HYPERPARAMS
'''
parameters = {
    'n_estimators': [100, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}
clf = ExtraTreesClassifier()
grid_search = GridSearchCV(clf,
                           parameters,
                           scoring='f1_macro',
                           n_jobs=6,
                           cv=3,
                           verbose=2)
grid_search.fit(X_train, y_train)

print('best estimator', grid_search.best_estimator_)
print('best params', grid_search.best_params_)

# %%
'''
Using best params from grid search...
best estimator ExtraTreesClassifier(class_weight='balanced',
                                    criterion='entropy',
                                    max_depth=10,
                                    max_features='sqrt',
                                    min_samples_leaf=5,
                                    min_samples_split=10,
                                    n_estimators=500)
best params {
    'class_weight': 'balanced',
    'criterion': 'entropy',
    'max_depth': 10,
    'max_features': 'sqrt',
    'min_samples_leaf': 5,
    'min_samples_split': 10,
    'n_estimators': 500
}
'''

best_params = {
    'class_weight': 'balanced',
    'criterion': 'entropy',
    'max_depth': 10,
    'max_features': 'sqrt',
    'min_samples_leaf': 5,
    'min_samples_split': 10,
    'n_estimators': 500
}
et_clf = ExtraTreesClassifier(**best_params)

et_clf.fit(X_train, y_train)

print('test accuracy')
y_hat = et_clf.predict(X_test)
print(classification_report(y_test, y_hat))

print('all towns')
y_all_hat = et_clf.predict(X_data)
print(classification_report(y.targets, y_all_hat))

selected_data['predicted'] = y_all_hat
# %%
print('predicted positive')
pp = []
for pop_id, r in selected_data[selected_data.predicted == 1].iterrows():
    print(pop_id, r.predicted, r.city_type, r.city_name)
    if r.city_name is not None:
        pp.append(r.city_name)
print(', '.join(sorted(set(pp))))
# %%
print('false positives')
for pop_id, r in selected_data.iterrows():
    if r.predicted == 1 and r.city_type not in town_classifications:
        print(pop_id, r.predicted, r.city_type, r.city_name)

print('false negatives')
for pop_id, r in selected_data.iterrows():
    if r.predicted == 0 and r.city_type in town_classifications:
        print(pop_id, r.predicted, r.city_type, r.city_name)

'''
test accuracy
              precision    recall  f1-score   support
           0       0.97      0.95      0.96        91
           1       0.87      0.92      0.89        36
    accuracy                           0.94       127
   macro avg       0.92      0.93      0.92       127
weighted avg       0.94      0.94      0.94       127
all towns
              precision    recall  f1-score   support
           0       0.99      0.98      0.99       451
           1       0.96      0.98      0.97       181
    accuracy                           0.98       632
   macro avg       0.98      0.98      0.98       632
weighted avg       0.98      0.98      0.98       632

predicted positive
19 1 New Town Bracknell
22 1 Expanded Town Like Luton
25 1 Expanded Town Farnborough
27 1 New Town Milton Keynes
28 1 New Town Cwmbran
30 1 New Town Northampton
34 1 Expanded Town Swindon
37 1 New Town Warrington
40 1 New Town Peterborough
42 1 Expanded Town Like Slough
43 1 Expanded Town Like Gloucester
44 1 New Town Telford
45 1 Expanded Town Burnley
48 1 Expanded Town Like Rotherham
59 1 New Town Like High Wycombe
61 1 New Town Crawley
63 1 New Town Basildon
66 1 Expanded Town Basingstoke
72 1 Expanded Town Hastings (& Bexhill)
73 1 New Town Stevenage
74 1 New Town Washington
76 1 New Town Like Wath Upon Dearne & environs
79 1 New Town Hemel Hempstead
80 1 New Town Like Port Talbot & Neath
82 1 New Town Like Nuneaton
83 1 New Town Like Bracknell
88 1 New Town Harlow
89 1 Expanded Town Tamworth
91 1 New Town Like Grays
92 1 Expanded Town Aylesbury
103 1 Expanded Town Ashford
106 1 Expanded Town Like Newbury
107 1 Expanded Town Like Bognor Regis
109 1 New Town Like Ellesmere Port
113 1 New Town Runcorn
114 1 Expanded Town Like Taunton
115 1 New Town Like Deeside
118 1 Expanded Town Like Widnes
120 1 New Town Like Bridgend
121 1 Expanded Town Like Eastleigh
123 1 New Town Corby
126 1 New Town Redditch
127 1 Expanded Town Like Stafford
128 1 New Town Like Castleford & Normanton
129 1 New Town Like Kidderminster
130 1 New Town Like Barry & Gibbons Down
131 1 Expanded Town Like Littlehampton
136 1 Expanded Town Like Horsham
137 1 Expanded Town Like Clacton On Sea
141 1 Expanded Town Braintree
142 1 Expanded Town Wellingborough
143 1 Expanded Town Banbury
146 1 New Town Like Sittingbourne
147 1 New Town Welwyn Garden City
150 1 Expanded Town Like Yeovil
153 1 New Town Like Cambourne & Redruth
155 1 Expanded Town Kings Lynn
156 1 Expanded Town Like Bromsgrove
158 1 Expanded Town Andover
161 1 Expanded Town Letchworth (Garden City)
162 1 New Town Like Kirkby
166 1 New Town Like Bexhill
169 1 Expanded Town Grantham
170 1 Expanded Town Like Northwich
171 1 Expanded Town Bury St Edmunds
172 1 New Town Like Trowbridge
173 1 New Town Like Swadlincote
174 1 New Town Skelmersdale
178 1 New Town Like Locks Heath
180 1 New Town Like Leyland
183 1 New Town Like Tonbridge
184 1 New Town Like Canvey Island
187 1 New Town Hatfield
189 1 Expanded Town Like Leighton Buzzard
190 1 Expanded Town Like Rushden
192 1 Expanded Town Like Chippenham
195 1 New Town Like Abingdon
196 1 None None
198 1 New Town Like Downside
199 1 New Town Like Cramlington
200 1 New Town Like Yate
202 1 New Town Like Haywards Heath
206 1 New Town Like Bicester
216 1 New Town Like Addlestone
217 1 New Town Like Hammonds Green
218 1 Expanded Town Like Burgess Hill
226 1 Expanded Town Like Felixstowe
227 1 New Town Like Burntwood
231 1 New Town Like Stanford-le-Hope
234 1 New Town Peterlee
236 1 New Town Like Winsford
237 1 New Town Newton Aycliffe
241 1 Expanded Town Haverhill
247 1 New Town Like Maghull
248 1 New Town Like Didcot
251 1 New Town Like Larkfield
252 1 Expanded Town Witham
254 1 New Town Like Plymstock
255 1 New Town Like Daventry
258 1 Expanded Town Thetford
261 1 New Town Like Coalville
264 1 New Town Like Lowton
266 1 New Town Like Moons Moat South
267 1 New Town Like Rugeley
269 1 Expanded Town Huntingdon
270 1 New Town Like Droitwich
280 1 Expanded Town Like Chapeltown
281 1 New Town Like Hoyland
284 1 New Town Like Winshill
289 1 New Town Like New Addington
292 1 Expanded Town Like Horley
298 1 Expanded Town Sudbury
301 1 New Town Like Clevedon
305 1 New Town Like Hailsham
314 1 None Harwich
316 1 Expanded Town Gainsborough
318 1 New Town Like Mid Norfolk
321 1 Expanded Town Like Tiverton
324 1 Expanded Town Like Stowmarket
331 1 None None
339 1 Expanded Town St Neots
341 1 Expanded Town Like Melksham
344 1 New Town Like Carcroft
352 1 New Town Like Fair Oak
357 1 Expanded Town Like Calne
371 1 New Town Like Shirehampton & Avonmouth
376 1 New Town Like South Woodham Ferrers
379 1 New Town Like Bordon
383 1 New Town Like Swanley
384 1 None None
385 1 New Town Like Hipswell
389 1 New Town Like Carterton
390 1 New Town Like Ryton & Crawcrook
391 1 New Town Like Baughurst
392 1 Expanded Town Like Keynsham
395 1 New Town Like Swallownest
396 1 Expanded Town Like Rochford
398 1 New Town Like Broadmeadows
407 1 New Town Like Kidlington
408 1 New Town Like Taverham
420 1 Expanded Town Bodmin
423 1 None None
428 1 New Town Like Leabrooks
435 1 None None
442 1 New Town Like Hadfield & Brookfield
448 1 New Town Like Wymondham
449 1 New Town Like Bishops Cleeve
454 1 New Town Like Westwells & Pickwick
455 1 New Town Like Wombourne
459 1 New Town Like Brackley
466 1 New Town Like Newhaven Town
470 1 New Town Like Bargoed
473 1 New Town Like Eaton Ford
478 1 New Town Like Belah
484 1 New Town Like Broadway & Littlemoor
486 1 New Town Like Euxton
490 1 New Town Like Brough
491 1 New Town Like Stocksbridge
497 1 New Town Like Cinderford
501 1 New Town Like Morton
503 1 Expanded Town Sandy
506 1 New Town Like Farncombe
507 1 New Town Like Mount Sorrel & Environs
508 1 Expanded Town Like Honiton
530 1 New Town Like Lee-on-the-Solent
536 1 New Town Like Vickerstown
541 1 None None
546 1 Expanded Town Like Kingsteignton
551 1 New Town Like Shepton Mallet
553 1 New Town Like Snodland
556 1 Expanded Town Like Amesbury
561 1 Expanded Town Like Llandudno Junction
565 1 New Town Like Chadwell St Mary
568 1 New Town Like West Ravendale
585 1 New Town Like Brandon & Meadowfield
594 1 New Town Like Seaton Delaval
596 1 New Town Like Killamarsh
601 1 New Town Like Tidworth
612 1 New Town Like Primethorpe & Broughton Astley
616 1 Expanded Town Mildenhall
623 1 New Town Like Southwater
626 1 New Town Like Dunmow
628 1 New Town Like Raunds
648 1 New Town Like Copperhouse
650 1 New Town Like Cambourne

false positives
196 1 None None
314 1 None Harwich
331 1 None None
384 1 None None
423 1 None None
435 1 None None
541 1 None None

false negatives
26 0 Expanded Town Plymouth
265 0 New Town Like Westhoughton
532 0 New Town Newtown
'''
