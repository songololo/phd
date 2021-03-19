Deprecated data processing steps
================================

Old roadnodes table creation methods
------------------------------------

1. Create a roadnode table for the corresponding roadlink table. This SQL statement will fetch the start and end point
   of each edge and use these as a basis for a new roadnodes table. Because some nodes are shared between edges, the
   results are collapsed (aggregated) based on the unique nodes. During this process, the corresponding out-edges for
   each node are aggregated in the edges column.

> This is a generic method for deducing roadnodes. OS provides their own roadnodes info, which isn't used here so as to keep the workflow generic. It also prevents potential confusion with start vs. end sides of lines.

```postgresql
CREATE TABLE analysis.roadnodes AS
  -- insert new node ids based on row numbers
  -- note that new table is aggregated based on matching nodes, and edge ids are aggregated accordingly
  SELECT row_number() OVER (ORDER BY node) as id, node as geom, city_id, array_agg(id) as edges FROM
    -- get the start and end point of each roadlink, then union and use as a basis for the road nodes table
    (SELECT ST_StartPoint(geom) as node, city_id, id FROM analysis.roadlinks
    UNION
    SELECT ST_EndPoint(geom) as node, city_id, id FROM analysis.roadlinks)
    AS nodes
  -- aggregate by node, therefore eliminating duplicates and aggregating corresponding edges for each node
  GROUP BY node, city_id;
-- completed in 1m 13s 667ms
-- add indices
CREATE UNIQUE INDEX IF NOT EXISTS id_idx_roadnodes ON analysis.roadnodes (id);
-- completed in 1s 872ms
CREATE INDEX IF NOT EXISTS edges_idx_roadnodes ON analysis.roadnodes USING GIN (edges);
-- completed in 2m 16s 534ms
CREATE INDEX IF NOT EXISTS city_id_idx_roadnodes ON analysis.roadnodes (city_id);
-- completed in 8s 138ms
CREATE INDEX IF NOT EXISTS geom_idx_roadnodes ON analysis.roadnodes USING GIST (geom);
-- completed in 32s 132ms
VACUUM ANALYSE analysis.roadnodes;
```

1. Add a neighbour_nodes column to the roadnodes table and generate each node's neighbouring nodes

> Make sure you've created an index on the edges column before proceeding (see above for index)! Even so, this will take about a day to run. The benefit to doing this is that you don't need to create a graph or calculate neighbouring nodes when running centrality algorithms.

```postgresql
ALTER TABLE analysis.roadnodes
    ADD COLUMN neighbour_nodes bigint[];
-- completed in 3ms
UPDATE analysis.roadnodes
    SET neighbour_nodes = agg_nodes.neighbours
    FROM
      (SELECT r_n_1.id as id, array_agg(r_n_2.id) as neighbours
          FROM analysis.roadnodes as r_n_1,
            analysis.roadnodes as r_n_2
          -- array overlap operator &&
          WHERE r_n_1.edges && r_n_2.edges
            AND r_n_1.id != r_n_2.id
          GROUP BY r_n_1.id
      ) as agg_nodes
    WHERE id = agg_nodes.id;
-- 1662223 rows affected in 22h 17m 6s 136ms
```

Old OS roads import methods
---------------------------

1. Import the OS Open Roads network.

- The downloaded data consists of separate BNG grid tiles. These were opened and merged in QGIS.
  > Located at /Users/gareth/Dropbox/Data/PhD/OS_open/2016_11_open_roads
- The tiles were then transferred to the VM using Dropbox and `wget`.
- `shp2pgsql` was used to convert the data to `sql`:  
  `shp2pgsql -I -s 27700 -c merged_os_open_roads mres.os_open_roads > os_open_roads.sql`
- The data can then be imported to postgres using `psql`, however, particular connection formatting is required for
  SSL:  
  `psql "sslmode=require host=localhost port=5432 dbname=data" --username=platplaas --file=os_open_roads.sql`

2. Import OS Address Base Premium addresses:

* Convert directory of gml.zip to shapefile using the following bash script. Run the script from the data directory.

  ```bash
  #!/bin/bash
  mkdir shp_output
  for FILE in *.zip # cycles through all files in directory (case-sensitive!)
  do
      echo "converting file: $FILE..."
      unzip $FILE # unzip to gml
      FILEGML=`echo $FILE | sed "s/_gml.zip/.gml/"` # gml file name
      FILEGFS=`echo $FILEGML | sed "s/.gml/.gfs/"` # for removing gfs files
      FILESHP=`echo $FILEGML | sed "s/.gml/.shp/"` # target file name
      /Applications/Postgres.app/Contents/Versions/9.6/bin/ogr2ogr \
      -f "ESRI Shapefile" \
      shp_output/$FILESHP $FILEGML
      rm $FILEGML $FILEGFS
  done
  exit
  ```

* ABORTED: Convert to sql and import using psql from the command line. NOTE: This didn't work because of varchar import
  errors for subsequently imported tables...

  ```bash
    #!/bin/bash
    
    /Applications/Postgres.app/Contents/Versions/9.6/bin/shp2pgsql -c -e -s 27700 -I `ls *.shp | sort -n | head -1` public.addresses | \
        /Applications/Postgres.app/Contents/Versions/9.6/bin/psql -h localhost -d my_db -U my_username
      
    for f in *.shp; do 
        if [ ! $f == `ls *.shp | sort -n | head -1` ]; then
            /Applications/Postgres.app/Contents/Versions/9.6/bin/shp2pgsql -a -e -s 27700 -I $f public.addresses | \
                /Applications/Postgres.app/Contents/Versions/9.6/bin/psql -h localhost -d my_db -U my_username
        fi
    done
  ```

* NEXT ATTEMPT: Load all the shapefiles into QGIS and use the merge functionality. Save a single shapefile for streets
  and another file for addresses.

* Cleanup columns, keeping USRN, UPRN, land use classification codes

3. Import VOA ->

4. Import the city boundaries

5. Create the 10, 20, 40, 80m segmented roads

```psql

```

Old stats aggregation method directly on DB:
-------------------------------------------

```postgresql
-- THIS IS DEPRECATED
ALTER TABLE analysis.roadnodes_20
    ADD COLUMN IF NOT EXISTS cens_tot_pop_interp real,
    ADD COLUMN IF NOT EXISTS cens_nocars_interp real,
    ADD COLUMN IF NOT EXISTS cens_cars_interp real,
    ADD COLUMN IF NOT EXISTS cens_ttw_peds_interp real,
    ADD COLUMN IF NOT EXISTS cens_ttw_bike_interp real,
    ADD COLUMN IF NOT EXISTS cens_ttw_motors_interp real,
    ADD COLUMN IF NOT EXISTS cens_ttw_pubtrans_interp real,
    ADD COLUMN IF NOT EXISTS cens_ttw_home_interp real,
    ADD COLUMN IF NOT EXISTS cens_dwellings_interp real,
    ADD COLUMN IF NOT EXISTS cens_students_interp real;
VACUUM ANALYSE analysis.roadnodes_20;
UPDATE analysis.roadnodes_20
    SET cens_tot_pop_interp = result.pop_idw,
        cens_nocars_interp = result.nocars_idw,
        cens_cars_interp = result.cars_idw,
        cens_ttw_peds_interp = result.foot_idw,
        cens_ttw_bike_interp = result.bike_idw,
        cens_ttw_motors_interp = result.motor_idw,
        cens_ttw_pubtrans_interp = result.pubtrans_idw,
        cens_ttw_home_interp = result.home_idw,
        cens_dwellings_interp = result.dwellings_idw,
        cens_students_interp = result.students_idw
    FROM
      (SELECT nodes.id,
          sum(calc.w_totpop) / sum(calc.d) AS pop_idw,
          sum(calc.w_nocars) / sum(calc.d) AS nocars_idw,
          sum(calc.w_cars) / sum(calc.d) AS cars_idw, 
          sum(calc.w_foot) / sum(calc.d) AS foot_idw,
          sum(calc.w_bike) / sum(calc.d) AS bike_idw,
          sum(calc.w_motor) / sum(calc.d) AS motor_idw,
          sum(calc.w_pubtrans) / sum(calc.d) AS pubtrans_idw,
          sum(calc.w_home) / sum(calc.d) AS home_idw,
          sum(calc.w_dwellings) / sum(calc.d) AS dwellings_idw,
          sum(calc.w_students) / sum(calc.d) AS students_idw
      FROM
        -- only run for cities to save some time?
        (SELECT id, geom FROM analysis.roadnodes_20
          WHERE city_id::int < 100) AS nodes
        CROSS JOIN LATERAL
        (SELECT geom,
            totpop,
            nocars,
            car1 + car2 + car3 + car4plus as cars,
            ttwfoot,
            ttwbike,
            ttwcar + ttwcarpass + ttwmbike + ttwtaxi as ttwmotor, 
            ttwbus + ttwtube + ttwtrain as ttwpubtrans,
            ttwhome,
            dwelling,
            allstudent,
            geom <-> nodes.geom AS dist 
          FROM census_2011.census_centroids
          -- threshold distance adds time but avoids issues e.g. at rivers or parks
          -- but would likely result in some cliff-edges?
          WHERE ST_DWithin(geom, nodes.geom, 300)
          ORDER BY geom <-> nodes.geom LIMIT 3) AS census
        CROSS JOIN LATERAL
        (SELECT
          totpop / dist^2 as w_totpop,
          nocars / dist^2 as w_nocars,
          cars / dist^2 AS w_cars,
          ttwfoot / dist^2 AS w_foot,
          ttwbike / dist^2 AS w_bike,
          ttwmotor / dist^2 AS w_motor,
          ttwpubtrans / dist^2 AS w_pubtrans,
          ttwhome / dist^2 AS w_home,
          dwelling / dist^2 AS w_dwellings,
          allstudent / dist^2 AS w_students,
          1 / dist^2 as d) AS calc
        GROUP BY id
      ) as result
    WHERE analysis.roadnodes_20.id = result.id;
VACUUM ANALYSE analysis.roadnodes_20;
-- 20m 28282463 rows affected in 6h 32m 2s 186ms
-- 50m 12011421 rows affected in 2h 6m 47s 395ms
-- 100m 6666145 rows affected in 1h 8m 27s 36ms
```