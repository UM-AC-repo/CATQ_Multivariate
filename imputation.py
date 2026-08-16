#!/usr/bin/env python3
"""
Hierarchy-preserving imputation for the UM autism questionnaire dataset.

Core principles
---------------
1. Never impute a derived total independently from its components.
2. When a component is missing, impute/derive the component first and then
   recompute the total.
3. Preserve observed values; do not silently "repair" contradictory records.
4. Derive AQ10_cutoff from a valid AQ10 total, or constrain an imputed AQ10
   total to the already-observed cutoff category.
5. Create multiple completed datasets rather than treating one fill-in as
   observed truth.

Dataset-specific safeguard
--------------------------
The supplied AQ10 current-subscore columns fail their summation identity in
164 of 187 fully observed rows. They are therefore quarantined by default and
excluded from completed outputs. The AQ10 total is imputed at total level,
conditional on the observed cutoff, until the original AQ10 items/subscores
can be rebuilt correctly. AQ10_max is internally coherent, so its missing
subscores are imputed first and its total is recomputed passively.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


AQ10_COMPONENTS = [
    "AQ10_att-details",
    "AQ10_att-switching",
    "AQ10_comm",
    "AQ10_imag",
    "AQ10_soc-skills",
]

AQ10_MAX_COMPONENTS = [
    "AQ10max_att-details",
    "AQ10max_att-switching",
    "AQ10max_comm",
    "AQ10max_imag",
    "AQ10max_soc-skills",
]

CATQ_COMPONENTS = ["CAT-Q_comp", "CAT-Q_mask", "CAT-Q_assim"]

CORE_AUXILIARIES = [
    "sex",
    "age",
    "EQ",
    "SQ",
    "CAT-Q",
    "CAT-Q_comp",
    "CAT-Q_mask",
    "CAT-Q_assim",
    "AQ10_cutoff",
    "AQ10",
    "AQ10_max",
    *AQ10_MAX_COMPONENTS,
]

ZSCORE_COLUMNS = [
    "EQ",
    "SQ",
    "CAT-Q",
    "AQ10",
    "AQ10_max",
    *CATQ_COMPONENTS,
    *AQ10_MAX_COMPONENTS,
]


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV/TSV/TXT using delimiter inference, preserving the raw file."""
    return pd.read_csv(path, sep=None, engine="python")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Required columns are absent: {missing}")


def identity_summary(
    df: pd.DataFrame, total: str, components: list[str]
) -> dict[str, object]:
    complete = df[total].notna() & df[components].notna().all(axis=1)
    component_sum = df[components].sum(axis=1, min_count=len(components))
    equal = pd.Series(False, index=df.index)
    equal.loc[complete] = np.isclose(
        df.loc[complete, total].astype(float),
        component_sum.loc[complete].astype(float),
    )
    inconsistent = complete & ~equal
    return {
        "total": total,
        "components": components,
        "n_complete_comparisons": int(complete.sum()),
        "n_consistent": int(equal.sum()),
        "n_inconsistent": int(inconsistent.sum()),
        "inconsistent_ids": df.loc[inconsistent, "ID"].astype(int).tolist(),
    }


def robust_scale(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return 1.0
    iqr = float(clean.quantile(0.75) - clean.quantile(0.25))
    if iqr > 0:
        return iqr
    sd = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    return sd if sd > 0 else 1.0


def donor_distances(
    target: pd.Series,
    donors: pd.DataFrame,
    features: list[str],
) -> pd.Series:
    """
    Robust standardized Euclidean distance using only features observed in
    both the target record and a donor record.
    """
    available = [
        feature
        for feature in features
        if feature in donors.columns
        and feature in target.index
        and pd.notna(target[feature])
        and donors[feature].notna().any()
    ]
    if not available:
        return pd.Series(0.0, index=donors.index)

    squared = pd.DataFrame(index=donors.index)
    for feature in available:
        scale = robust_scale(donors[feature])
        squared[feature] = (
            (pd.to_numeric(donors[feature], errors="coerce") - float(target[feature]))
            / scale
        ) ** 2
    return np.sqrt(squared.mean(axis=1, skipna=True)).fillna(np.inf)


def draw_near_donor(
    target: pd.Series,
    donors: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
    k: int,
) -> pd.Series:
    if donors.empty:
        raise ValueError(f"No eligible donors for participant ID {target['ID']}")
    distance = donor_distances(target, donors, features)
    eligible = distance.replace([np.inf, -np.inf], np.nan).dropna().nsmallest(
        min(k, len(distance))
    )
    if eligible.empty:
        raise ValueError(f"No finite donor distance for participant ID {target['ID']}")

    zero = eligible[np.isclose(eligible, 0.0)]
    if not zero.empty:
        chosen_index = rng.choice(zero.index.to_numpy())
    else:
        weights = 1.0 / (eligible.to_numpy(dtype=float) + 1e-8)
        weights = weights / weights.sum()
        chosen_index = rng.choice(eligible.index.to_numpy(), p=weights)
    return donors.loc[chosen_index]


def impute_aq10_total(
    completed: pd.DataFrame,
    original: pd.DataFrame,
    rng: np.random.Generator,
    k: int,
) -> None:
    """
    Impute missing AQ10 totals from observed-score donors in the same cutoff
    category. This is a fallback because the supplied current AQ10 subscores
    are structurally inconsistent and cannot safely drive the total.
    """
    missing_rows = original.index[original["AQ10"].isna()]
    donor_base = original.loc[original["AQ10"].notna()].copy()
    for index in missing_rows:
        target = completed.loc[index]
        cutoff = target["AQ10_cutoff"]
        donors = donor_base.loc[donor_base["AQ10_cutoff"] == cutoff]
        donor = draw_near_donor(
            target,
            donors,
            [
                column
                for column in CORE_AUXILIARIES
                if column not in {"AQ10", "AQ10_cutoff"}
            ],
            rng,
            k,
        )
        value = int(round(float(donor["AQ10"])))
        if int(cutoff) == 1 and value < 6:
            raise AssertionError("Positive-cutoff AQ10 donor was below 6")
        if int(cutoff) == 0 and value >= 6:
            raise AssertionError("Negative-cutoff AQ10 donor was 6 or above")
        completed.at[index, "AQ10"] = value


def impute_aq10_max_components(
    completed: pd.DataFrame,
    original: pd.DataFrame,
    rng: np.random.Generator,
    k: int,
) -> None:
    """
    Fill missing AQ10_max components, then passively recompute AQ10_max.

    If a total is observed and exactly one component is absent, use deductive
    imputation. Otherwise draw all missing components from one nearby complete
    donor to retain their joint observed pattern.
    """
    complete_donors = original.loc[
        original[AQ10_MAX_COMPONENTS].notna().all(axis=1)
    ].copy()
    observed_bounds = {
        column: (
            int(original[column].dropna().min()),
            int(original[column].dropna().max()),
        )
        for column in AQ10_MAX_COMPONENTS
    }

    affected = original.index[
        original[AQ10_MAX_COMPONENTS].isna().any(axis=1)
        | original["AQ10_max"].isna()
    ]
    for index in affected:
        missing_components = [
            column
            for column in AQ10_MAX_COMPONENTS
            if pd.isna(original.at[index, column])
        ]

        if not missing_components:
            completed.at[index, "AQ10_max"] = int(
                round(completed.loc[index, AQ10_MAX_COMPONENTS].sum())
            )
            continue

        if original.at[index, "AQ10_max"] == original.at[index, "AQ10_max"]:
            if len(missing_components) == 1:
                column = missing_components[0]
                remainder = float(original.at[index, "AQ10_max"]) - float(
                    completed.loc[
                        index,
                        [c for c in AQ10_MAX_COMPONENTS if c != column],
                    ].sum()
                )
                lower, upper = observed_bounds[column]
                if (
                    not math.isclose(remainder, round(remainder))
                    or remainder < lower
                    or remainder > upper
                ):
                    raise ValueError(
                        f"Deductive value {remainder} for {column}, ID "
                        f"{int(original.at[index, 'ID'])}, is outside [{lower}, {upper}]"
                    )
                completed.at[index, column] = int(round(remainder))
            else:
                known_sum = float(
                    completed.loc[
                        index,
                        [
                            c
                            for c in AQ10_MAX_COMPONENTS
                            if c not in missing_components
                        ],
                    ].sum()
                )
                needed = int(round(float(original.at[index, "AQ10_max"]) - known_sum))
                donors = complete_donors.loc[
                    complete_donors[missing_components].sum(axis=1) == needed
                ]
                donor = draw_near_donor(
                    completed.loc[index],
                    donors,
                    [
                        c
                        for c in CORE_AUXILIARIES
                        if c not in set(missing_components + ["AQ10_max"])
                    ],
                    rng,
                    k,
                )
                for column in missing_components:
                    completed.at[index, column] = int(donor[column])
        else:
            donor = draw_near_donor(
                completed.loc[index],
                complete_donors,
                [
                    c
                    for c in CORE_AUXILIARIES
                    if c not in set(missing_components + ["AQ10_max"])
                ],
                rng,
                k,
            )
            for column in missing_components:
                completed.at[index, column] = int(donor[column])

        completed.at[index, "AQ10_max"] = int(
            round(completed.loc[index, AQ10_MAX_COMPONENTS].sum())
        )


def add_imputation_flags(completed: pd.DataFrame, original: pd.DataFrame) -> None:
    tracked = ["AQ10", "AQ10_max", *AQ10_MAX_COMPONENTS]
    flag_columns = []
    for column in tracked:
        flag = f"{column}_imputed"
        completed[flag] = original[column].isna().astype(int)
        flag_columns.append(flag)
    completed["any_imputed"] = completed[flag_columns].max(axis=1)
    completed["AQ10_subscores_quarantined"] = 1


def add_zscores(completed: pd.DataFrame) -> None:
    for column in ZSCORE_COLUMNS:
        if column not in completed.columns:
            continue
        values = pd.to_numeric(completed[column], errors="coerce")
        sd = float(values.std(ddof=1))
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"Cannot standardize constant/invalid column {column}")
        completed[f"{column}_z"] = (values - float(values.mean())) / sd


def validate_completed(completed: pd.DataFrame) -> dict[str, object]:
    required = [
        "AQ10",
        "AQ10_cutoff",
        "AQ10_max",
        *AQ10_MAX_COMPONENTS,
        "CAT-Q",
        *CATQ_COMPONENTS,
    ]
    n_missing = int(completed[required].isna().sum().sum())

    catq_sum = completed[CATQ_COMPONENTS].sum(axis=1)
    aqmax_sum = completed[AQ10_MAX_COMPONENTS].sum(axis=1)
    cutoff = (completed["AQ10"].astype(float) >= 6).astype(int)

    catq_ok = bool(np.isclose(completed["CAT-Q"], catq_sum).all())
    aqmax_ok = bool(np.isclose(completed["AQ10_max"], aqmax_sum).all())
    cutoff_ok = bool((completed["AQ10_cutoff"].astype(int) == cutoff).all())

    if n_missing or not catq_ok or not aqmax_ok or not cutoff_ok:
        raise AssertionError(
            {
                "n_missing_required": n_missing,
                "catq_identity_ok": catq_ok,
                "aq10_max_identity_ok": aqmax_ok,
                "aq10_cutoff_ok": cutoff_ok,
            }
        )
    return {
        "n_missing_required": n_missing,
        "catq_identity_ok": catq_ok,
        "aq10_max_identity_ok": aqmax_ok,
        "aq10_cutoff_ok": cutoff_ok,
    }


def run(
    input_path: Path,
    output_dir: Path,
    m: int,
    seed: int,
    k: int,
) -> dict[str, object]:
    raw = read_table(input_path)
    require_columns(
        raw,
        [
            "ID",
            "AQ10_cutoff",
            "AQ10",
            "AQ10_max",
            "CAT-Q",
            *AQ10_COMPONENTS,
            *AQ10_MAX_COMPONENTS,
            *CATQ_COMPONENTS,
        ],
    )
    if raw["ID"].duplicated().any():
        duplicates = raw.loc[raw["ID"].duplicated(keep=False), "ID"].tolist()
        raise ValueError(f"Duplicate participant IDs: {duplicates}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_missing = raw.isna()
    summaries = {
        "source_file": input_path.name,
        "n_rows": int(len(raw)),
        "missing_cells_before": int(source_missing.sum().sum()),
        "rows_with_missing_before": raw.loc[
            source_missing.any(axis=1), "ID"
        ].astype(int).tolist(),
        "identity_before": {
            "CAT-Q": identity_summary(raw, "CAT-Q", CATQ_COMPONENTS),
            "AQ10": identity_summary(raw, "AQ10", AQ10_COMPONENTS),
            "AQ10_max": identity_summary(raw, "AQ10_max", AQ10_MAX_COMPONENTS),
        },
        "quarantined_columns": AQ10_COMPONENTS,
        "reason_for_quarantine": (
            "Observed AQ10 total and current-subscore sum disagree in the "
            "large majority of fully observed rows; observed values were not overwritten."
        ),
        "imputations": [],
    }

    if summaries["identity_before"]["CAT-Q"]["n_inconsistent"]:
        raise ValueError("CAT-Q identity is inconsistent before imputation")
    if summaries["identity_before"]["AQ10_max"]["n_inconsistent"]:
        raise ValueError("AQ10_max identity is inconsistent before imputation")

    audit_rows = []
    for imputation in range(1, m + 1):
        rng = np.random.default_rng(seed + imputation - 1)
        completed = raw.copy(deep=True)

        impute_aq10_total(completed, raw, rng, k)
        impute_aq10_max_components(completed, raw, rng, k)

        # Current AQ10 subscores are excluded rather than silently overwritten.
        completed = completed.drop(columns=AQ10_COMPONENTS)
        add_imputation_flags(completed, raw)
        add_zscores(completed)
        validation = validate_completed(completed)

        output_name = f"UM_autism_total_level_imputation_{imputation:02d}.csv"
        output_path = output_dir / output_name
        completed.to_csv(output_path, index=False)

        for index in raw.index[raw.isna().any(axis=1)]:
            row = {
                "imputation": imputation,
                "ID": int(raw.at[index, "ID"]),
            }
            for column in ["AQ10", "AQ10_max", *AQ10_MAX_COMPONENTS]:
                if pd.isna(raw.at[index, column]):
                    row[column] = completed.at[index, column]
            audit_rows.append(row)

        summaries["imputations"].append(
            {
                "number": imputation,
                "file": output_name,
                **validation,
            }
        )

    pd.DataFrame(audit_rows).to_csv(
        output_dir / "UM_autism_imputation_draws_audit.csv", index=False
    )
    with (output_dir / "UM_autism_imputation_qc.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summaries, handle, indent=2)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Raw CSV/TSV/TXT dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("UM_autism_hierarchical_imputation"),
    )
    parser.add_argument(
        "--m",
        type=int,
        default=20,
        help="Number of completed datasets (default: 20)",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest eligible donors (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(
        input_path=arguments.input,
        output_dir=arguments.output_dir,
        m=arguments.m,
        seed=arguments.seed,
        k=arguments.k,
    )
    print(
        f"Created {len(result['imputations'])} completed datasets in "
        f"{arguments.output_dir}"
    )
