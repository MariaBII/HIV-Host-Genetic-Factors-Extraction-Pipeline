import csv
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment


# Evaluation configuration
RESULTS_DIR = "LLM_results_100"
GOLD_FILE = "gold_dataset_100.csv"
OUTPUT_DIR = "LLM_evaluation_100"

NULL_VALUES = {
    "", "none", "null", "nan", "n/a", "na",
    "not provided", "not reported"
}

FIELDS = [
    "genetic_factor",
    "factor_type",
    "association",
    "population",
]

FIELD_LABELS = {
    "genetic_factor": "Host genetic factor",
    "factor_type": "Factor type",
    "association": "Association type",
    "population": "Population",
}

# Fuzzy-matching thresholds
THRESHOLDS = {
    "genetic_factor": 70.0,
    "factor_type": 60.0,
    "association": 60.0,
    "population": 60.0,
}

MODEL_ORDER = [
    "GPT-3.5 Turbo",
    "GPT-4 Turbo",
    "GPT-4.1",
    "GPT-4o",
    "GPT-5",
    "Claude Haiku",
    "Claude Sonnet",
    "Claude Opus",
]

INPUT_ORDER = ["abstract", "full_text"]

INPUT_LABELS = {
    "abstract": "Abstract",
    "full_text": "Full text",
}

FILE_MAP = {
    ("GPT-3.5 Turbo", "abstract"):
        "extracted_factors_100_abstracts_GPT3.5_Turbo.csv",
    ("GPT-3.5 Turbo", "full_text"):
        "extracted_factors_100_full-text_GPT_3.5_Turbo.csv",

    ("GPT-4 Turbo", "abstract"):
        "extracted_factors_100_abstracts_GPT4_Turbo.csv",
    ("GPT-4 Turbo", "full_text"):
        "extracted_factors_100_full-text_GPT_4_Turbo.csv",

    ("GPT-4.1", "abstract"):
        "extracted_factors_100_abstracts_GPT4.1.csv",
    ("GPT-4.1", "full_text"):
        "extracted_factors_100_full-text_GPT_4.1.csv",

    ("GPT-4o", "abstract"):
        "extracted_factors_100_abstracts_GPT4o.csv",
    ("GPT-4o", "full_text"):
        "extracted_factors_100_full-text_GPT-4o.csv",

    ("GPT-5", "abstract"):
        "extracted_factors_100_abstracts_GPT5.csv",
    ("GPT-5", "full_text"):
        "extracted_factors_100_full-text_GPT5.csv",

    ("Claude Haiku", "abstract"):
        "extracted_factors_100_abstracts_Claude_Haiku.csv",
    ("Claude Haiku", "full_text"):
        "extracted_factors_100_full-text_Claude_Haiku.csv",

    ("Claude Sonnet", "abstract"):
        "extracted_factors_100_abstracts_Claude_Sonnet.csv",
    ("Claude Sonnet", "full_text"):
        "extracted_factors_100_full-text_Claude_Sonnet.csv",

    ("Claude Opus", "abstract"):
        "extracted_factors_100_abstracts_Claude_Opus.csv",
    ("Claude Opus", "full_text"):
        "extracted_factors_100_full-text_Claude_Opus.csv",
}


RS_RE = re.compile(r"\brs\d+\b", re.IGNORECASE)

HLA_RE = re.compile(
    r"HLA[\s-]*([A-Z0-9]+)\*([0-9]{2,3})"
    r"(?::([0-9]{2,3}))?(?:/([0-9:/]+))?",
    re.IGNORECASE,
)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def clean_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in NULL_VALUES:
        return ""

    return text


# Light normalisation is used only for comparison during evaluation.
# It does not replace the separate biological normalisation used later
# for the large-scale extraction dataset.
def norm_text(value: object) -> str:
    text = clean_text(value)

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text).lower()
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )

    return re.sub(r"\s+", " ", text).strip()


def norm_factor(value: object) -> str:
    text = norm_text(value)

    if not text:
        return ""

    text = text.replace("Δ", "delta").replace("δ", "delta")

    return re.sub(r"\s+", "", text)


def fuzzy_similarity(a: object, b: object) -> float:
    left = norm_text(a)
    right = norm_text(b)

    if not left or not right:
        return 0.0

    return float(fuzz.token_set_ratio(left, right))


def extract_hla_alleles(value: object) -> Set[str]:
    text = norm_text(value).upper()
    alleles: Set[str] = set()

    for match in HLA_RE.finditer(text):
        locus, group, field2, remainder = match.groups()

        if field2:
            alleles.add(f"HLA-{locus}*{group}:{field2}")

            if remainder:
                for part in remainder.split("/"):
                    if not part:
                        continue

                    if ":" in part:
                        first, second = part.split(":", 1)
                        alleles.add(
                            f"HLA-{locus}*{first}:{second}"
                        )
                    else:
                        alleles.add(
                            f"HLA-{locus}*{group}:{part}"
                        )

        else:
            alleles.add(f"HLA-{locus}*{group}")

            if remainder:
                for part in remainder.split("/"):
                    if part:
                        alleles.add(f"HLA-{locus}*{part}")

    return alleles


def extract_rs_alleles(value: object) -> Dict[str, Set[str]]:
    text = norm_text(value)

    identifiers: Dict[str, Set[str]] = {
        item.lower(): set()
        for item in RS_RE.findall(text)
    }

    for match in re.finditer(
        r"\b(rs\d+)\b(.{0,30})",
        text,
        re.IGNORECASE,
    ):
        rsid = match.group(1).lower()
        tail = match.group(2)

        allele_match = re.search(
            r"(?:allele|genotype)?\s*[:=\(\[]*\s*"
            r"([ACGT](?:[/|>][ACGT])?|[ACGT]{2,3})\b",
            tail,
            re.IGNORECASE,
        )

        if allele_match:
            token = (
                allele_match.group(1)
                .upper()
                .replace("|", "/")
                .replace(">", "/")
            )

            identifiers.setdefault(rsid, set()).update(
                token.split("/")
            )

    return identifiers


def normalise_allele_set(values: Iterable[str]) -> Set[str]:
    alleles: Set[str] = set()

    for value in values:
        token = value.upper()

        if all(character in "ACGT" for character in token):
            alleles.update(token)

        else:
            for part in re.split(r"[/|>]", token):
                if part and all(
                    character in "ACGT"
                    for character in part
                ):
                    alleles.update(part)

    return alleles


def hla_compatible(left: Set[str], right: Set[str]) -> bool:
    for a in left:
        for b in right:
            if (
                a == b
                or a.startswith(b + ":")
                or b.startswith(a + ":")
            ):
                return True

    return False


def extract_special_variants(value: object) -> Set[str]:
    raw = clean_text(value)
    variants: Set[str] = set()

    if re.search(
        r"(?:delta|Δ|δ)[-\s]*32|"
        r"(?:del(?:etion)?)[-\s]*32|"
        r"32[-\s]*bp\s*deletion",
        raw,
        re.IGNORECASE,
    ):
        variants.add("delta32")

    return variants


def factor_similarity(a: object, b: object) -> float:
    left = norm_factor(a)
    right = norm_factor(b)

    if not left or not right:
        return 0.0

    if left == right:
        return 100.0

    left_rs = extract_rs_alleles(a)
    right_rs = extract_rs_alleles(b)

    left_rsids = set(left_rs)
    right_rsids = set(right_rs)

    left_hla = extract_hla_alleles(a)
    right_hla = extract_hla_alleles(b)

    left_special = extract_special_variants(a)
    right_special = extract_special_variants(b)

    # Explicitly different rsIDs are not allowed to match.
    if left_rsids and right_rsids:
        common = left_rsids & right_rsids

        if not common:
            return 0.0

        for rsid in common:
            left_alleles = normalise_allele_set(
                left_rs.get(rsid, set())
            )
            right_alleles = normalise_allele_set(
                right_rs.get(rsid, set())
            )

            if (
                left_alleles
                and right_alleles
                and not left_alleles.intersection(
                    right_alleles
                )
            ):
                return 0.0

        return max(95.0, fuzzy_similarity(a, b))

    if bool(left_rsids) != bool(right_rsids):
        return min(60.0, fuzzy_similarity(a, b))

    # Explicitly conflicting HLA alleles are not allowed to match.
    if left_hla and right_hla:
        if not hla_compatible(left_hla, right_hla):
            return 0.0

        return max(95.0, fuzzy_similarity(a, b))

    if bool(left_hla) != bool(right_hla):
        return min(60.0, fuzzy_similarity(a, b))

    if left_special and right_special:
        if not left_special.intersection(right_special):
            return 0.0

        return max(95.0, fuzzy_similarity(a, b))

    if bool(left_special) != bool(right_special):
        return min(60.0, fuzzy_similarity(a, b))

    return max(
        float(fuzz.ratio(left, right)),
        fuzzy_similarity(a, b),
    )


def deduplicate_annotations(
    rows: Sequence[dict],
) -> Tuple[List[dict], List[dict]]:
    unique: List[dict] = []
    duplicates: List[dict] = []

    observed: Set[
        Tuple[str, str, str, str, str]
    ] = set()

    for row in rows:
        key = (
            row["paper_id"],
            norm_factor(row["genetic_factor"]),
            norm_text(row["population"]),
            norm_text(row["factor_type"]),
            norm_text(row["association"]),
        )

        if key in observed:
            duplicates.append(row)

        else:
            observed.add(key)
            unique.append(row)

    return unique, duplicates


def load_gold(
    path: Path,
) -> Tuple[Set[str], List[dict], List[dict]]:
    source = read_csv(path)

    paper_ids: Set[str] = set()
    annotations: List[dict] = []

    for row_number, row in enumerate(source, start=2):
        paper_id = clean_text(row.get("paper_id"))

        if not paper_id:
            raise ValueError(
                f"Gold-standard row {row_number} "
                "has no paper_id"
            )

        paper_ids.add(paper_id)

        factor = clean_text(
            row.get("host_genetic_factor")
        )

        if not factor:
            continue

        annotations.append(
            {
                "source_row": row_number,
                "paper_id": paper_id,
                "genetic_factor": factor,
                "population": clean_text(
                    row.get("population")
                ),
                "factor_type": clean_text(
                    row.get("factor_type")
                ),
                "association": clean_text(
                    row.get("association")
                ),
                "quote": clean_text(row.get("quote")),
            }
        )

    annotations, duplicates = deduplicate_annotations(
        annotations
    )

    return paper_ids, annotations, duplicates


def load_predictions(
    path: Path,
) -> Tuple[Set[str], List[dict], List[dict]]:
    source = read_csv(path)

    paper_ids: Set[str] = set()
    annotations: List[dict] = []

    for row_number, row in enumerate(source, start=2):
        paper_id = clean_text(row.get("paper_id"))

        if paper_id:
            paper_ids.add(paper_id)

        factor = clean_text(
            row.get("genetic_factor")
        )

        if not factor:
            continue

        annotations.append(
            {
                "source_row": row_number,
                "paper_id": paper_id,
                "genetic_factor": factor,
                "population": clean_text(
                    row.get("population")
                ),
                "factor_type": clean_text(
                    row.get("factor_type")
                ),
                "association": clean_text(
                    row.get("association")
                ),
                "quote": clean_text(row.get("quote")),
            }
        )

    annotations, duplicates = deduplicate_annotations(
        annotations
    )

    return paper_ids, annotations, duplicates


def match_annotations(
    gold_rows: Sequence[dict],
    predicted_rows: Sequence[dict],
) -> Tuple[List[tuple], List[int], List[int]]:

    n_gold = len(gold_rows)
    n_pred = len(predicted_rows)

    if n_gold == 0 or n_pred == 0:
        return (
            [],
            list(range(n_gold)),
            list(range(n_pred)),
        )

    # Dummy rows/columns allow annotations to remain unmatched.
    size = n_gold + n_pred

    score_matrix = np.zeros(
        (size, size),
        dtype=float,
    )

    score_matrix[:n_gold, :n_pred] = -1e9

    pair_details: Dict[
        Tuple[int, int],
        Tuple[float, float, float, float],
    ] = {}

    for gold_index, gold in enumerate(gold_rows):
        for pred_index, prediction in enumerate(
            predicted_rows
        ):
            genetic_score = factor_similarity(
                gold["genetic_factor"],
                prediction["genetic_factor"],
            )

            if (
                genetic_score
                < THRESHOLDS["genetic_factor"]
            ):
                continue

            factor_type_score = fuzzy_similarity(
                gold["factor_type"],
                prediction["factor_type"],
            )

            association_score = fuzzy_similarity(
                gold["association"],
                prediction["association"],
            )

            population_score = fuzzy_similarity(
                gold["population"],
                prediction["population"],
            )

            # Genetic identity dominates; contextual fields
            # resolve competing candidate matches.
            total_score = (
                genetic_score * 1000.0
                + factor_type_score
                + association_score
                + population_score
            )

            score_matrix[
                gold_index,
                pred_index,
            ] = total_score

            pair_details[
                (gold_index, pred_index)
            ] = (
                genetic_score,
                factor_type_score,
                association_score,
                population_score,
            )

    assigned_rows, assigned_columns = (
        linear_sum_assignment(-score_matrix)
    )

    matches: List[tuple] = []

    for gold_index, pred_index in zip(
        assigned_rows,
        assigned_columns,
    ):
        if (
            gold_index < n_gold
            and pred_index < n_pred
            and score_matrix[
                gold_index,
                pred_index,
            ] > 0
        ):
            matches.append(
                (
                    gold_index,
                    pred_index,
                    pair_details[
                        (gold_index, pred_index)
                    ],
                )
            )

    matched_gold = {
        gold_index
        for gold_index, _, _ in matches
    }

    matched_pred = {
        pred_index
        for _, pred_index, _ in matches
    }

    unmatched_gold = [
        index
        for index in range(n_gold)
        if index not in matched_gold
    ]

    unmatched_pred = [
        index
        for index in range(n_pred)
        if index not in matched_pred
    ]

    return (
        matches,
        unmatched_gold,
        unmatched_pred,
    )


def compare_field(
    gold_value: object,
    predicted_value: object,
    threshold: float,
) -> Tuple[int, int, int, float]:

    gold = clean_text(gold_value)
    prediction = clean_text(predicted_value)

    if not gold and not prediction:
        return 0, 0, 0, 0.0

    if gold and not prediction:
        return 0, 0, 1, 0.0

    if not gold and prediction:
        return 0, 1, 0, 0.0

    similarity = fuzzy_similarity(
        gold,
        prediction,
    )

    if similarity >= threshold:
        return 1, 0, 0, similarity

    return 0, 1, 1, similarity


def calculate_metrics(
    tp: int,
    fp: int,
    fn: int,
) -> Tuple[float, float, float]:

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


def evaluate_model(
    model: str,
    input_type: str,
    path: Path,
    gold_ids: Set[str],
    gold_annotations: Sequence[dict],
) -> Tuple[List[dict], dict, List[dict], dict]:

    source_rows = read_csv(path)

    prediction_ids, predictions, duplicate_predictions = (
        load_predictions(path)
    )

    gold_by_paper: MutableMapping[
        str,
        List[dict],
    ] = defaultdict(list)

    pred_by_paper: MutableMapping[
        str,
        List[dict],
    ] = defaultdict(list)

    for row in gold_annotations:
        gold_by_paper[row["paper_id"]].append(row)

    for row in predictions:
        pred_by_paper[row["paper_id"]].append(row)

    counts = {
        field: Counter(TP=0, FP=0, FN=0)
        for field in FIELDS
    }

    audit: List[dict] = []
    matched_factor_scores: List[float] = []

    for paper_id in sorted(gold_ids):
        gold_rows = gold_by_paper[paper_id]
        predicted_rows = pred_by_paper[paper_id]

        (
            matches,
            unmatched_gold,
            unmatched_pred,
        ) = match_annotations(
            gold_rows,
            predicted_rows,
        )

        for (
            gold_index,
            pred_index,
            pair_scores,
        ) in matches:

            gold = gold_rows[gold_index]
            prediction = predicted_rows[pred_index]
            genetic_score = pair_scores[0]

            matched_factor_scores.append(
                genetic_score
            )

            counts[
                "genetic_factor"
            ]["TP"] += 1

            audit.append(
                {
                    "model": model,
                    "input_type": input_type,
                    "paper_id": paper_id,
                    "status": "matched",
                    "field": "genetic_factor",
                    "gold_value": gold[
                        "genetic_factor"
                    ],
                    "predicted_value": prediction[
                        "genetic_factor"
                    ],
                    "similarity": genetic_score,
                    "TP": 1,
                    "FP": 0,
                    "FN": 0,
                }
            )

            for field in (
                "factor_type",
                "association",
                "population",
            ):
                (
                    tp,
                    fp,
                    fn,
                    score,
                ) = compare_field(
                    gold[field],
                    prediction[field],
                    THRESHOLDS[field],
                )

                counts[field]["TP"] += tp
                counts[field]["FP"] += fp
                counts[field]["FN"] += fn

                audit.append(
                    {
                        "model": model,
                        "input_type": input_type,
                        "paper_id": paper_id,
                        "status": (
                            "match"
                            if tp
                            else "mismatch"
                            if fp and fn
                            else "missing_or_extra"
                        ),
                        "field": field,
                        "gold_value": gold[field],
                        "predicted_value":
                            prediction[field],
                        "similarity": score,
                        "TP": tp,
                        "FP": fp,
                        "FN": fn,
                    }
                )

        for gold_index in unmatched_gold:
            gold = gold_rows[gold_index]

            for field in FIELDS:
                gold_value = (
                    gold["genetic_factor"]
                    if field == "genetic_factor"
                    else gold[field]
                )

                if not clean_text(gold_value):
                    continue

                counts[field]["FN"] += 1

                audit.append(
                    {
                        "model": model,
                        "input_type": input_type,
                        "paper_id": paper_id,
                        "status": "unmatched_gold",
                        "field": field,
                        "gold_value": gold_value,
                        "predicted_value": "",
                        "similarity": 0.0,
                        "TP": 0,
                        "FP": 0,
                        "FN": 1,
                    }
                )

        for pred_index in unmatched_pred:
            prediction = predicted_rows[pred_index]

            for field in FIELDS:
                predicted_value = (
                    prediction["genetic_factor"]
                    if field == "genetic_factor"
                    else prediction[field]
                )

                if not clean_text(predicted_value):
                    continue

                counts[field]["FP"] += 1

                audit.append(
                    {
                        "model": model,
                        "input_type": input_type,
                        "paper_id": paper_id,
                        "status":
                            "unmatched_prediction",
                        "field": field,
                        "gold_value": "",
                        "predicted_value":
                            predicted_value,
                        "similarity": 0.0,
                        "TP": 0,
                        "FP": 1,
                        "FN": 0,
                    }
                )

    # Predictions assigned to papers outside the
    # 100-paper gold standard are false positives.
    for paper_id in sorted(
        set(pred_by_paper) - gold_ids
    ):
        for prediction in pred_by_paper[paper_id]:
            for field in FIELDS:
                predicted_value = (
                    prediction["genetic_factor"]
                    if field == "genetic_factor"
                    else prediction[field]
                )

                if clean_text(predicted_value):
                    counts[field]["FP"] += 1

    per_field: List[dict] = []

    for field in FIELDS:
        tp = counts[field]["TP"]
        fp = counts[field]["FP"]
        fn = counts[field]["FN"]

        precision, recall, f1 = (
            calculate_metrics(tp, fp, fn)
        )

        per_field.append(
            {
                "model": model,
                "input_type": input_type,
                "field": field,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

    total_tp = sum(
        counts[field]["TP"]
        for field in FIELDS
    )

    total_fp = sum(
        counts[field]["FP"]
        for field in FIELDS
    )

    total_fn = sum(
        counts[field]["FN"]
        for field in FIELDS
    )

    precision, recall, f1 = calculate_metrics(
        total_tp,
        total_fp,
        total_fn,
    )

    overall = {
        "model": model,
        "input_type": input_type,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    validation = {
        "model": model,
        "input_type": input_type,
        "source_file": path.name,
        "source_rows": len(source_rows),
        "non_null_rows_after_deduplication":
            len(predictions),
        "duplicate_rows_removed":
            len(duplicate_predictions),
        "unique_source_paper_ids":
            len(prediction_ids),
        "missing_gold_paper_ids":
            len(gold_ids - prediction_ids),
        "extra_paper_ids":
            len(prediction_ids - gold_ids),
        "matched_genetic_factors":
            counts["genetic_factor"]["TP"],
        "mean_matched_factor_similarity":
            statistics.mean(matched_factor_scores)
            if matched_factor_scores
            else 0.0,
    }

    return (
        per_field,
        overall,
        audit,
        validation,
    )


def create_heatmap(
    per_field_rows: Sequence[dict],
    output_dir: Path,
) -> None:

    lookup = {
        (
            row["model"],
            row["input_type"],
            row["field"],
        ): row["F1"]
        for row in per_field_rows
    }

    row_pairs = [
        (model, input_type)
        for model in MODEL_ORDER
        for input_type in INPUT_ORDER
    ]

    values = np.array(
        [
            [
                lookup[
                    (model, input_type, field)
                ]
                for field in FIELDS
            ]
            for model, input_type in row_pairs
        ]
    )

    labels = [
        f"{model} - {INPUT_LABELS[input_type]}"
        for model, input_type in row_pairs
    ]

    figure, axis = plt.subplots(
        figsize=(9.5, 10.0)
    )

    image = axis.imshow(
        values,
        aspect="auto",
        cmap="YlGnBu",
        vmin=values.min(),
        vmax=values.max(),
    )

    axis.set_xticks(
        np.arange(len(FIELDS)),
        [
            FIELD_LABELS[field]
            for field in FIELDS
        ],
    )

    axis.set_yticks(
        np.arange(len(labels)),
        labels,
    )

    axis.set_xlabel("Extraction field")
    axis.set_ylabel("Model and input type")

    for row_index in range(values.shape[0]):
        for column_index in range(
            values.shape[1]
        ):
            axis.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=11,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.035,
        pad=0.025,
    )

    colorbar.set_label("F1-score")

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "100_paper_per_field_F1_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_overall_f1_figure(
    overall_rows: Sequence[dict],
    output_dir: Path,
) -> None:

    lookup = {
        (
            row["model"],
            row["input_type"],
        ): row["F1"]
        for row in overall_rows
    }

    y_positions = np.arange(
        len(MODEL_ORDER)
    )

    bar_height = 0.36

    abstract_values = [
        lookup[(model, "abstract")]
        for model in MODEL_ORDER
    ]

    full_text_values = [
        lookup[(model, "full_text")]
        for model in MODEL_ORDER
    ]

    figure, axis = plt.subplots(
        figsize=(9.5, 6.5)
    )

    abstract_bars = axis.barh(
        y_positions - bar_height / 2,
        abstract_values,
        height=bar_height,
        label="Abstract",
        hatch="//",
    )

    full_text_bars = axis.barh(
        y_positions + bar_height / 2,
        full_text_values,
        height=bar_height,
        label="Full text",
    )

    axis.bar_label(
        abstract_bars,
        labels=[f"{value:.3f}" for value in abstract_values],
        padding=3,
        fontsize=8,
    )

    axis.bar_label(
        full_text_bars,
        labels=[f"{value:.3f}" for value in full_text_values],
        padding=3,
        fontsize=8,
    )

    axis.set_yticks(
        y_positions,
        MODEL_ORDER,
    )

    axis.invert_yaxis()
    axis.set_xlim(0, 1.05)

    axis.set_xlabel(
        "Overall micro-averaged F1-score"
    )

    axis.legend(frameon=False)

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "100_paper_overall_F1.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    results_dir = script_dir / RESULTS_DIR
    gold_path = script_dir / GOLD_FILE
    output_dir = script_dir / OUTPUT_DIR

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results folder not found: "
            f"{results_dir}"
        )

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold-standard file not found: "
            f"{gold_path}"
        )

    missing_files = [
        filename
        for filename in FILE_MAP.values()
        if not (
            results_dir / filename
        ).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing LLM result files:\n"
            + "\n".join(missing_files)
        )

    gold_ids, gold_annotations, duplicate_gold = (
        load_gold(gold_path)
    )

    if len(gold_ids) != 100:
        raise ValueError(
            "Expected 100 gold-standard "
            f"paper IDs, found {len(gold_ids)}"
        )

    all_per_field: List[dict] = []
    all_overall: List[dict] = []
    all_audit: List[dict] = []
    all_validation: List[dict] = []

    for model in MODEL_ORDER:
        for input_type in INPUT_ORDER:
            path = (
                results_dir
                / FILE_MAP[(model, input_type)]
            )

            (
                per_field,
                overall,
                audit,
                validation,
            ) = evaluate_model(
                model,
                input_type,
                path,
                gold_ids,
                gold_annotations,
            )

            all_per_field.extend(per_field)
            all_overall.append(overall)
            all_audit.extend(audit)
            all_validation.append(validation)

    write_csv(
        output_dir
        / "llm_100_per_field_metrics.csv",
        all_per_field,
    )

    write_csv(
        output_dir
        / "llm_100_global_micro_metrics.csv",
        all_overall,
    )

    write_csv(
        output_dir
        / "llm_100_match_audit.csv",
        all_audit,
    )

    write_csv(
        output_dir
        / "llm_100_validation_report.csv",
        all_validation,
    )

    create_heatmap(
        all_per_field,
        output_dir,
    )

    create_overall_f1_figure(
        all_overall,
        output_dir,
    )


if __name__ == "__main__":
    main()