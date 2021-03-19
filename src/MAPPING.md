- Create the inner selection bounds

```postgresql
INSERT INTO boundaries.london_samples VALUES (
    4,
    ST_SetSRID(ST_MakeBox2D(ST_Point(528072, 178907), ST_Point(534245, 183589)), 27700),
    'Inner London Selection'
);
```

- If creating a temporary table:

```postgresql
create table analysis.temp_20 as 
  select * from analysis.roadnodes_20 
    where ST_Intersects(ST_SetSRID(ST_MakeBox2D(ST_Point(528072, 178907), ST_Point(534245, 183589)), 27700), geom);
ALTER TABLE analysis.temp_20 ADD PRIMARY KEY (id);

create table analysis.temp_20_dual as select * from analysis.roadnodes_20_dual where ST_Intersects(ST_SetSRID(ST_MakeBox2D(ST_Point(528072, 178907), ST_Point(534245, 183589)), 27700), geom);
ALTER TABLE analysis.temp_20_dual ADD PRIMARY KEY (id);

create table analysis.temp_poi as select * from os.poi where ST_Intersects(ST_SetSRID(ST_MakeBox2D(ST_Point(528072, 178907), ST_Point(534245, 183589)), 27700), geom) and date_last_updated = date('2018-06-01');

ALTER TABLE analysis.temp_poi ADD PRIMARY KEY (urn);


select count(poi.urn) from
   (select geom from analysis.city_boundaries_150 where pop_id = 1) as boundary,
   (select urn, geom from os.poi where date_last_updated = date('2018-06-01')) as poi
where ST_Intersects(boundary.geom, poi.geom);

```

- plot actual vs. randomised mixed uses:

```postgresql
select lu.*
    from analysis.roadnodes_20_lu as lu,
         (select geom from boundaries.london_samples where id = 4) as bounds
    where ST_Intersects(bounds.geom, lu.geom);

select rdm_lu.*
    from analysis.roadnodes_20_lu_randomised as rdm_lu,
         (select geom from boundaries.london_samples where id = 4) as bounds
    where ST_Intersects(bounds.geom, rdm_lu.geom);

-- create diff columns on full table
CREATE TABLE analysis.roadnodes_20_lu_diff AS
SELECT id, geom, mu_hill_branch_wt_0_100, mu_hill_branch_wt_0_400, mu_hill_branch_wt_0_1600
FROM analysis.roadnodes_20_lu
WHERE city_pop_id = 1;

ALTER TABLE analysis.roadnodes_20_lu_diff
    ADD COLUMN rdm_100 real,
    ADD COLUMN rdm_400 real,
    ADD COLUMN rdm_1600 real,
    ADD COLUMN diff_100 real,
    ADD COLUMN diff_400 real,
    ADD COLUMN diff_1600 real;

UPDATE analysis.roadnodes_20_lu_diff AS lu_diff
SET
    rdm_100 = lu_rdm.mu_hill_branch_wt_0_100,
    rdm_400 = lu_rdm.mu_hill_branch_wt_0_400,
    rdm_1600 = lu_rdm.mu_hill_branch_wt_0_1600
FROM analysis.roadnodes_20_lu_randomised AS lu_rdm WHERE lu_diff.id = lu_rdm.id;

UPDATE analysis.roadnodes_20_lu_diff
    SET diff_100 = mu_hill_branch_wt_0_100 - rdm_100,
        diff_400 = mu_hill_branch_wt_0_400 - rdm_400,
        diff_1600 = mu_hill_branch_wt_0_1600 - rdm_1600;

ALTER TABLE analysis.roadnodes_20_lu_diff ADD PRIMARY KEY (id);
CREATE INDEX IF NOT EXISTS geom_idx_roadnodes_20_lu_diff ON analysis.roadnodes_20_lu_diff USING GIST (geom);
```
