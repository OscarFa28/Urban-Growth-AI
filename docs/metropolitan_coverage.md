# Metropolitan Coverage Registry

Introduced in v0.26.0, the metropolitan coverage registry defines the municipal coverage expected for each project city or metropolitan area. It is a configuration layer for validating territorial scope before rebuilding boundaries, grids, features, modeling datasets, or inference datasets.

The current v0.26.0 work does not change the default behavior of the existing city, grid, feature, training, or scoring scripts. Rebuilding metropolitan grids and downstream datasets is reserved for v0.27.0.

## Source

The intended source of truth for Mexico is Metrópolis de México 2020 from SEDATU, CONAPO, and INEGI. The seed file is small and versioned at:

```bash
configs/metropolis/mx_metropolitan_municipalities_2020.csv
```

Rows marked as `partial_manual_review_required` or `standalone_manual_review_required` are placeholders for validation and workflow integration. They must be reviewed against the official metropolitan municipality list before they are used to rebuild spatial datasets.

## Columns

- `country_code`: ISO-like country code, normalized to uppercase.
- `metro_area_id`: Stable project ID for the metropolitan or local coverage unit.
- `metro_area_name`: Human-readable name.
- `city_id`: Existing project city ID from `configs/cities/mexico.yaml`.
- `city_name`: Human-readable project city name.
- `municipality_cvegeo`: Municipality or demarcation CVEGEO. It is loaded as text to preserve leading zeros.
- `municipality_name`: Municipality or demarcation name.
- `state_name`: State name.
- `state_code`: State code, loaded as text.
- `is_core_municipality`: Boolean-like text flag for the seed/core municipality.
- `source_name`: Source label.
- `source_year`: Source year.
- `coverage_status`: Current review status for the row or coverage unit.
- `notes`: Manual notes for incomplete or standalone coverage.

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

The validator checks required columns, empty IDs, duplicate `metro_area_id + municipality_cvegeo` pairs, compact counts by metropolitan area, and current city IDs without coverage.

## v0.27.0 Pending Work

For v0.27.0, replace the seed rows with the reviewed official 2020 municipality membership, then rebuild metropolitan boundaries, grids, features, modeling datasets, inference datasets, and scores using the complete coverage.
