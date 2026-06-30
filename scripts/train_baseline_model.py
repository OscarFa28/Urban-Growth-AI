"""Train baseline urban growth model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from joblib import dump

from urban_growth.modeling.baseline import create_logistic_regression_baseline
from urban_growth.modeling.training import (
    TARGET_COLUMN,
    evaluate_binary_classifier,
    find_best_f1_threshold,
    predict_scores,
    prepare_modeling_data,
    read_modeling_dataset,
    summarize_split,
    temporal_train_validation_test_split,
)


def default_modeling_dataset_path(
    country_code: str,
    grid_size: int,
    start_year: int,
    end_year: int,
    priorities: list[int],
) -> Path:
    country = country_code.lower()

    if len(priorities) == 1:
        label = f"priority_{priorities[0]}"
    else:
        label = "priorities_" + "_".join(str(priority) for priority in priorities)

    return (
        Path("data/features/modeling")
        / country
        / f"{grid_size}m"
        / f"{country}_modeling_dataset_{start_year}_{end_year}_{label}_{grid_size}m.parquet"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline model.")
    parser.add_argument("--country-code", default="MX")
    parser.add_argument("--priorities", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--train-start-year", type=int, default=2016)
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--validation-year", type=int, default=2023)
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/metrics/baseline_logistic_regression_metrics.json"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/baselines/baseline_logistic_regression.joblib"),
    )
    parser.add_argument(
        "--include-urban-cells",
        action="store_true",
        help="Train with all cells instead of filtering to non-urban candidate cells.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print()
    print(title)
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()

    input_path = args.input or default_modeling_dataset_path(
        country_code=args.country_code,
        grid_size=args.grid_size,
        start_year=args.start_year,
        end_year=args.end_year,
        priorities=args.priorities,
    )

    print(f"Reading modeling dataset: {input_path}")
    raw_frame = read_modeling_dataset(input_path)

    prepared = prepare_modeling_data(
        raw_frame,
        target_column=TARGET_COLUMN,
        candidate_only=not args.include_urban_cells,
    )

    split = temporal_train_validation_test_split(
        prepared.frame,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        validation_year=args.validation_year,
        test_year=args.test_year,
    )

    train_features = prepared.features.loc[split.train.index]
    train_target = prepared.target.loc[split.train.index]

    validation_features = prepared.features.loc[split.validation.index]
    validation_target = prepared.target.loc[split.validation.index]

    test_features = prepared.features.loc[split.test.index]
    test_target = prepared.target.loc[split.test.index]

    print(f"Candidate only: {not args.include_urban_cells}")
    print(f"Feature count: {len(prepared.feature_columns):,}")
    print(f"Train rows: {len(train_features):,}")
    print(f"Validation rows: {len(validation_features):,}")
    print(f"Test rows: {len(test_features):,}")

    model = create_logistic_regression_baseline(
        train_features,
        random_state=args.random_state,
    )

    print("Training logistic regression baseline...")
    model.fit(train_features, train_target)

    validation_scores = predict_scores(model, validation_features)
    threshold = find_best_f1_threshold(validation_target, validation_scores)

    train_scores = predict_scores(model, train_features)
    test_scores = predict_scores(model, test_features)

    metrics = {
        "model": "logistic_regression",
        "target": TARGET_COLUMN,
        "candidate_only": not args.include_urban_cells,
        "feature_count": len(prepared.feature_columns),
        "feature_columns": prepared.feature_columns,
        "threshold": threshold,
        "splits": {
            "train": summarize_split(split.train),
            "validation": summarize_split(split.validation),
            "test": summarize_split(split.test),
        },
        "metrics": {
            "train": evaluate_binary_classifier(
                train_target,
                train_scores,
                threshold,
            ),
            "validation": evaluate_binary_classifier(
                validation_target,
                validation_scores,
                threshold,
            ),
            "test": evaluate_binary_classifier(
                test_target,
                test_scores,
                threshold,
            ),
        },
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    with args.metrics_output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    dump(
        {
            "model": model,
            "threshold": threshold,
            "feature_columns": prepared.feature_columns,
            "target": TARGET_COLUMN,
            "candidate_only": not args.include_urban_cells,
        },
        args.model_output,
    )

    print(f"Saved metrics: {args.metrics_output}")
    print(f"Saved model: {args.model_output}")

    print_metrics("Validation metrics", metrics["metrics"]["validation"])
    print_metrics("Test metrics", metrics["metrics"]["test"])


if __name__ == "__main__":
    main()
