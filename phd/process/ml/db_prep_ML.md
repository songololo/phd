
Timeseries
----------

1. Run `poi_epochs_hdf5.py` to create the timeseries data.


Machine Learning
----------------

1. Use the `random_forest_tuning.py` and `random_forest.py` scripts to do ML

1. Difference the existing and predicted values, e.g.
```postgresql
ALTER TABLE ml.roadnodes_100_test
    ADD COLUMN uses_eating_200_difference real;
    
UPDATE ml.roadnodes_100_test
    SET uses_eating_200_difference = uses_eating_200_predict - uses_eating_200_002;
```

