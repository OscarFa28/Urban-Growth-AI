# Spatial Units Strategy

Urban Growth AI uses a hierarchical geospatial structure to keep the dataset scalable and comparable across cities, municipalities, metropolitan areas, states, and countries.

## Current Spatial Unit

The current pipeline works primarily at the municipality level.

Each city entry in the registry represents a base spatial unit used to:

- download or define a boundary
- generate multi-resolution grids
- attach geospatial features
- build machine learning datasets

Most Mexican entries currently represent municipalities. Some exceptions may exist, such as Ciudad de México, which is represented as a federal entity in the initial version.

## Hierarchy

The intended spatial hierarchy is:

cell -> spatial unit -> metropolitan area -> state -> country

Where:

- cell is the grid-level observation used for modeling
- spatial unit is the base administrative or geographic unit, usually a municipality
- metropolitan area is a grouping of multiple municipalities or spatial units
- state is the federal state or equivalent administrative region
- country is the national-level grouping

## Why municipality-first?

The project starts with municipalities because they are easier to process, validate, and scale.

Metropolitan areas will be built later by grouping multiple municipalities instead of replacing the municipality-level pipeline.

This allows the project to support both municipality-level analysis and metropolitan-level analysis without rebuilding the dataset.

## Future Metropolitan Area Strategy

Metropolitan areas will be represented as groups of spatial units.

Example:

metro_area_id: mx_zm_guadalajara
metro_area_name: Zona Metropolitana de Guadalajara
spatial_units:
  - mx_guadalajara
  - mx_zapopan
  - mx_tlaquepaque
  - mx_tonala
  - mx_tlajomulco
  - mx_el_salto

The initial registry may include provisional metro_area_id and metro_area_name values. Later versions should validate metropolitan area membership using official or curated datasets.

## Dataset Implications

Each grid cell should eventually include:

- cell_id
- grid_size_m
- spatial_unit_id
- spatial_unit_type
- municipality_id
- metro_area_id
- state
- country_code

This keeps the dataset flexible enough for:

- municipality-level modeling
- metropolitan-level aggregation
- state-level comparisons
- cross-country expansion
