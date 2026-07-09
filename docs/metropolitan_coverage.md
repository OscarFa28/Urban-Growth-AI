# Metropolitan Coverage Registry

The metropolitan coverage registry defines the municipal or demarcation coverage expected for each project city or metropolitan area. It is a configuration layer for validating territorial scope before rebuilding boundaries, grids, features, modeling datasets, or inference datasets.

v0.26.0 introduced the lateral registry and validation workflow. v0.27.0 expands the Mexico registry with official/manual-reviewed municipality membership from Metrópolis de México 2020 for the project cities currently in `configs/cities/mexico.yaml`.

This version still does not change the default behavior of the city, grid, feature, training, inference, or scoring scripts. Rebuilding metropolitan boundaries, grids, features, datasets, and scores is reserved for v0.28.0.

## Source

The source of truth for official metropolitan rows is Metrópolis de México 2020, published by SEDATU, CONAPO, and INEGI. The rows in this repository were derived from the datos.gob.mx CSV resource `municipios_tipologia.csv`, which provides municipality-level typification for the 2020 metropolitan delimitations.

Reference URLs:

- https://www.datos.gob.mx/dataset/metropolis_mexico_2020
- https://www.gob.mx/sedatu/documentos/metropolis-de-mexico-2020?state=published

The versioned project registry is:

```bash
configs/metropolis/mx_metropolitan_municipalities_2020.csv
```

Rows with `coverage_status=official_2020` come from the official metropolitan municipality list. Rows with `coverage_status=standalone_manual_review_required` are local project coverage units that were not found as metropolitan areas in the official source and should remain out of a metropolitan grouping until reviewed.

## Columns

- `country_code`: ISO-like country code, normalized to uppercase.
- `metro_area_id`: Stable project ID for the metropolitan or local coverage unit.
- `metro_area_name`: Human-readable metropolitan or local coverage name.
- `city_id`: Existing project city ID from `configs/cities/mexico.yaml`.
- `city_name`: Human-readable project city name.
- `municipality_cvegeo`: Municipality or demarcation CVEGEO, loaded as text and validated as five digits.
- `municipality_name`: Municipality or demarcation name.
- `state_name`: State name.
- `state_code`: Two-digit state code, loaded as text.
- `is_core_municipality`: `true` when the municipality is central/core according to the source typification or local seed.
- `coverage_status`: One of `official_2020`, `manual_review_required`, or `standalone_manual_review_required`.
- `source_name`: Source label.
- `source_year`: Source year.

## Validation

Validate the registry with:

```bash
python scripts/validate_metropolitan_coverage.py
```

Optionally write a JSON summary:

```bash
python scripts/validate_metropolitan_coverage.py \
  --output-summary /tmp/metropolitan_coverage_summary.json
```

The validator checks required columns, empty required values, five-digit `municipality_cvegeo`, allowed `coverage_status`, boolean-like `is_core_municipality`, duplicate `metro_area_id + municipality_cvegeo` pairs, compact counts by metropolitan area, cities requiring manual review, and current city IDs without coverage.

## v0.28.0 Pending Work

For v0.28.0, use this completed registry to rebuild metropolitan boundaries, grids, features, modeling datasets, inference datasets, and scores using the complete municipal coverage.
