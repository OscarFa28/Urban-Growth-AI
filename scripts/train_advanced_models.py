"""Train advanced urban growth models."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

from joblib import dump
from sklearn.utils.class_weight import compute_sample_weight

from urban_growth.modeling.advanced import create_advanced_model
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


class TrainingProgress:
    """Simple terminal progress indicator for long model fits."""

    def __init__(self, message: str, interval_seconds: int = 30) -> None:
        self.message = message
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def __enter__(self) -> TrainingProgress:
        self._started_at = time.monotonic()
        print(f"  Started: {self.message}", flush=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=1)

        elapsed = time.monotonic() - self._started_at
        print(
            f"  Finished: {self.message} | elapsed: {format_elapsed(elapsed)}",
            flush=True,
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            elapsed = time.monotonic() - self._started_at
            print(
                f"  Still running: {self.message} | elapsed: {format_elapsed(elapsed)}",
                flush=True,
            )


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as hh:mm:ss or mm:ss."""
    minutes, remaining_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

    return f"{minutes:02d}:{remaining_seconds:02d}"


def priority_label(priorities: list[int]) -> str:
    if len(priorities) == 1:
        return f"priority_{priorities[0]}"
    return "priorities_" + "_".join(str(priority) for priority in priorities)


def default_modeling_dataset_path(
    country_code: str,
    grid_size: int,
    start_year: int,
    end_year: int,
    priorities: list[int],
) -> Path:
    country = country_code.lower()
    label = priority_label(priorities)

    return (
        Path("data/features/modeling")
        / country
        / f"{grid_size}m"
        / f"{country}_modeling_dataset_{start_year}_{end_year}_{label}_{grid_size}m.parquet"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train advanced models.")
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
        "--models",
        nargs="+",
        choices=["hist_gradient_boosting", "random_forest"],
        default=["hist_gradient_boosting", "random_forest"],
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/metrics/advanced_model_metrics.json"),
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=Path("models/advanced"),
    )
    parser.add_argument(
        "--include-urban-cells",
        action="store_true",
        help="Train with all cells instead of filtering to non-urban candidate cells.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--progress-interval-seconds",
        type=int,
        default=30,
        help="Seconds between progress messages while a model is fitting.",
    )
    parser.add_argument(
        "--model-verbose",
        type=int,
        default=1,
        help="Verbosity passed to scikit-learn classifiers.",
    )
    return parser.parse_args()


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print()
    print(title)
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def train_one_model(
    model_name: str,
    train_features,
    train_target,
    validation_features,
    validation_target,
    test_features,
    test_target,
    feature_columns: list[str],
    candidate_only: bool,
    random_state: int,
    model_output_dir: Path,
    model_index: int,
    model_count: int,
    progress_interval_seconds: int,
    model_verbose: int,
) -> dict[str, Any]:
    """Train and evaluate one advanced model."""
    start_percent = int(((model_index - 1) / model_count) * 100)
    end_percent = int((model_index / model_count) * 100)

    print()
    print("=" * 100)
    print(f"Training model {model_index}/{model_count}: {model_name}")
    print(f"Overall progress: {start_percent}%")

    model = create_advanced_model(
        model_name,
        train_features,
        random_state=random_state,
        verbose=model_verbose,
    )

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=train_target,
    )

    with TrainingProgress(
        message=f"fitting {model_name}",
        interval_seconds=progress_interval_seconds,
    ):
        model.fit(
            train_features,
            train_target,
            classifier__sample_weight=sample_weight,
        )

    print(f"Overall progress: {end_percent}%")
    print("Scoring validation split...")

    validation_scores = predict_scores(model, validation_features)
    threshold = find_best_f1_threshold(validation_target, validation_scores)

    print("Scoring train and test splits...")
    train_scores = predict_scores(model, train_features)
    test_scores = predict_scores(model, test_features)

    model_metrics = {
        "model": model_name,
        "target": TARGET_COLUMN,
        "candidate_only": candidate_only,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "threshold": threshold,
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

    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_output = model_output_dir / f"{model_name}.joblib"

    dump(
        {
            "model": model,
            "threshold": threshold,
            "feature_columns": feature_columns,
            "target": TARGET_COLUMN,
            "candidate_only": candidate_only,
        },
        model_output,
    )

    model_metrics["model_output"] = str(model_output)

    print_metrics(
        f"{model_name} validation metrics",
        model_metrics["metrics"]["validation"],
    )
    print_metrics(
        f"{model_name} test metrics",
        model_metrics["metrics"]["test"],
    )
    print(f"Saved model: {model_output}")

    return model_metrics


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
    print(f"Models: {args.models}")

    results: dict[str, Any] = {
        "target": TARGET_COLUMN,
        "candidate_only": not args.include_urban_cells,
        "feature_count": len(prepared.feature_columns),
        "splits": {
            "train": summarize_split(split.train),
            "validation": summarize_split(split.validation),
            "test": summarize_split(split.test),
        },
        "models": {},
    }

    model_count = len(args.models)

    for model_index, model_name in enumerate(args.models, start=1):
        results["models"][model_name] = train_one_model(
            model_name=model_name,
            train_features=train_features,
            train_target=train_target,
            validation_features=validation_features,
            validation_target=validation_target,
            test_features=test_features,
            test_target=test_target,
            feature_columns=prepared.feature_columns,
            candidate_only=not args.include_urban_cells,
            random_state=args.random_state,
            model_output_dir=args.model_output_dir,
            model_index=model_index,
            model_count=model_count,
            progress_interval_seconds=args.progress_interval_seconds,
            model_verbose=args.model_verbose,
        )

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Saved metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
