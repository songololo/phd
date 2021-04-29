# PhD codebase

Gareth Simons' PhD codebase

__Detection and prediction of urban archetypes at the pedestrian scale: computational toolsets, morphological metrics, and machine learning methods.__

Completed in 2021, though this work was completed over the span of the preceding five years.

This work is also available through the accompanying papers released in the `arXiv` preprint repository:
- links pending.

### Dataset preparation:

Data is primarily derived from:
- _Ordnance Survey_ _Open Roads_
- _Ordnance Survey_ _Points of Interest_
- _Office for National Statistics_ census data

A series of scripts processes the files and saves the derived data to a PostGIS enabled Postgres database.

- Centralities: 