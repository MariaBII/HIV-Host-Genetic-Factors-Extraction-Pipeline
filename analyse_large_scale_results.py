import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Input file
INPUT_CSV = "extracted_factors_1200_fulltext_GPT5_normalised.csv"

# Output files
OUTPUT_DIR = Path("large_scale_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "large_scale_summary.csv"
DISTRIBUTION_CSV = OUTPUT_DIR / "distribution_summary.csv"
TOP_10_CSV = OUTPUT_DIR / "top_10_normalised_factors.csv"
TOP_20_CSV = OUTPUT_DIR / "top_20_normalised_factors.csv"
JOINT_COUNTS_CSV = OUTPUT_DIR / "effect_association_joint_counts.csv"
RECORD_BINS_CSV = OUTPUT_DIR / "records_per_article_bins.csv"

RECORDS_PER_ARTICLE_FIG = OUTPUT_DIR / "records_per_article.png"
ENTITY_CLASS_FIG = OUTPUT_DIR / "entity_class_distribution.png"
EFFECT_ASSOCIATION_FIG = OUTPUT_DIR / "effect_association_bubble.png"
TOP_20_FACTORS_FIG = OUTPUT_DIR / "top_20_normalised_factors.png"


NONE_LIKE = {"", "none", "null", "nan", "na"}


def clean_value(value):
    if value is None:
        return ""
    return str(value).strip()


def is_non_null_factor(value):
    return clean_value(value).lower() not in NONE_LIKE


def classify_entity(value):
    """Rule-based descriptive classification of extracted factor records."""
    text = clean_value(value)
    upper = text.upper()

    if not text:
        return "Other/complex"

    if "HLA" in upper:
        return "HLA allele/feature"

    if re.search(r"\bhaplotype\b|\bhaplogroup\b", text, flags=re.IGNORECASE):
        return "Haplotype"

    if re.search(
        r"\bgenotype\b|\bhomozyg|\bheterozyg|\bzygos",
        text,
        flags=re.IGNORECASE,
    ):
        return "Genotype/zygosity"

    if re.search(r"\brs\d+\b", text, flags=re.IGNORECASE):
        return "SNP/variant"

    if re.search(
        r"\b(?:SNP|polymorphism|variant|mutation|deletion|insertion|indel)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "SNP/variant"

    if re.search(r"\b[A-Z]\d+[A-Z]\b", text):
        return "SNP/variant"

    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._/-]*(?:\s+[A-Za-z0-9][A-Za-z0-9._/-]*)?",
        text,
    ):
        return "Gene/protein"

    return "Other/complex"


def normalise_effect(value):
    text = clean_value(value).lower()

    mapping = {
        "protective": "Protective",
        "risk": "Risk",
        "mixed": "Mixed",
        "unclear": "Unclear",
    }

    return mapping.get(text, "Unclear")


def normalise_association(value):
    text = clean_value(value).lower().replace("_", " ")

    mapping = {
        "progression": "Progression",
        "resistance": "Resistance",
        "susceptibility": "Susceptibility",
        "treatment response": "Treatment response",
        "unclear": "Unclear",
    }

    return mapping.get(text, "Unclear")


def save_summary_table(df, non_null):
    positive_papers = non_null["paper_id"].nunique()
    all_papers = df["paper_id"].nunique()
    null_papers = all_papers - positive_papers

    records_per_positive_paper = non_null.groupby("paper_id").size()

    raw_unique = non_null["genetic_factor"].nunique()
    normalised_unique = non_null["normalised_genetic_factor"].nunique()

    summary = pd.DataFrame(
        [
            ["Europe PMC full-text articles processed", all_papers],
            ["Total pipeline output rows", len(df)],
            ["Papers with ≥1 extracted host genetic factor", positive_papers],
            ["Papers without extracted host genetic factors", null_papers],
            ["Non-null host genetic factor extraction records", len(non_null)],
            ["Unique extracted factor names before normalisation", raw_unique],
            ["Unique factors after normalisation", normalised_unique],
            ["Mean extraction records per positive paper", round(records_per_positive_paper.mean(), 2)],
            ["Median extraction records per positive paper", int(records_per_positive_paper.median())],
            ["Maximum extraction records in one paper", int(records_per_positive_paper.max())],
        ],
        columns=["Metric", "Value"],
    )

    summary.to_csv(SUMMARY_CSV, index=False)

    return records_per_positive_paper


def make_records_per_article_figure(df, non_null):
    counts = (
        non_null.groupby("paper_id")
        .size()
        .reindex(df["paper_id"].drop_duplicates(), fill_value=0)
    )

    categories = [
        "0",
        "1",
        "2–3",
        "4–5",
        "6–10",
        "11–20",
        ">20",
    ]

    values = [
        int((counts == 0).sum()),
        int((counts == 1).sum()),
        int(((counts >= 2) & (counts <= 3)).sum()),
        int(((counts >= 4) & (counts <= 5)).sum()),
        int(((counts >= 6) & (counts <= 10)).sum()),
        int(((counts >= 11) & (counts <= 20)).sum()),
        int((counts > 20).sum()),
    ]

    total_articles = len(counts)
    percentages = [
        value / total_articles * 100
        for value in values
    ]

    distribution = pd.DataFrame(
        {
            "Extraction records per article": categories,
            "Articles": values,
            "Percentage (%)": [round(p, 1) for p in percentages],
        }
    )

    distribution.to_csv(RECORD_BINS_CSV, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(categories, values)

    ax.set_yscale("log")
    ax.set_xlabel("Host genetic factor extraction records per article")
    ax.set_ylabel("Number of articles (log scale)")

    for bar, count, percentage in zip(bars, values, percentages):
        ax.annotate(
            f"{count}\n({percentage:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(RECORDS_PER_ARTICLE_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_entity_analysis(non_null):
    non_null = non_null.copy()

    non_null["entity_class"] = non_null[
        "normalised_genetic_factor"
    ].apply(classify_entity)

    order = [
        "HLA allele/feature",
        "Gene/protein",
        "SNP/variant",
        "Other/complex",
        "Genotype/zygosity",
        "Haplotype",
    ]

    entity_counts = (
        non_null["entity_class"]
        .value_counts()
        .reindex(order, fill_value=0)
    )

    total = len(non_null)
    percentages = entity_counts / total * 100

    plot_counts = entity_counts.iloc[::-1]
    plot_percentages = percentages.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.barh(
        plot_counts.index,
        plot_counts.values,
    )

    ax.set_xlabel("Extraction records")
    ax.set_ylabel("Biological entity class")

    max_count = max(plot_counts.max(), 1)
    ax.set_xlim(0, max_count * 1.20)

    for bar, count, percentage in zip(
        bars,
        plot_counts.values,
        plot_percentages.values,
    ):
        ax.text(
            bar.get_width() + max_count * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)} ({percentage:.1f}%)",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(ENTITY_CLASS_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return non_null, entity_counts


def make_distribution_table(non_null, entity_counts):
    total = len(non_null)

    effect_order = ["Protective", "Risk", "Unclear", "Mixed"]
    association_order = [
        "Progression",
        "Resistance",
        "Susceptibility",
        "Treatment response",
        "Unclear",
    ]

    effect_counts = (
        non_null["effect_direction"]
        .value_counts()
        .reindex(effect_order, fill_value=0)
    )

    association_counts = (
        non_null["association_category"]
        .value_counts()
        .reindex(association_order, fill_value=0)
    )

    rows = []

    for category, count in entity_counts.items():
        rows.append(
            [
                "Biological entity class",
                category,
                int(count),
                round(count / total * 100, 1),
            ]
        )

    for category, count in effect_counts.items():
        rows.append(
            [
                "Reported effect direction",
                category,
                int(count),
                round(count / total * 100, 1),
            ]
        )

    for category, count in association_counts.items():
        rows.append(
            [
                "HIV association category",
                category,
                int(count),
                round(count / total * 100, 1),
            ]
        )

    table = pd.DataFrame(
        rows,
        columns=[
            "Dimension",
            "Category",
            "Count",
            "Percentage (%)",
        ],
    )

    table.to_csv(DISTRIBUTION_CSV, index=False)

    return effect_counts, association_counts


def make_effect_association_figure(non_null):
    effect_order = ["Protective", "Risk", "Mixed", "Unclear"]

    association_order = [
        "Susceptibility",
        "Progression",
        "Resistance",
        "Treatment response",
        "Unclear",
    ]

    joint = pd.crosstab(
        non_null["effect_direction"],
        non_null["association_category"],
    ).reindex(
        index=effect_order,
        columns=association_order,
        fill_value=0,
    )

    joint.to_csv(JOINT_COUNTS_CSV)

    total = len(non_null)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    max_count = joint.to_numpy().max()
    scale = 1800 / max_count if max_count else 1

    for y, effect in enumerate(effect_order):
        for x, association in enumerate(association_order):
            count = int(joint.loc[effect, association])

            if count == 0:
                continue

            percentage = count / total * 100

            ax.scatter(
                x,
                y,
                s=max(count * scale, 55),
                alpha=0.75,
            )

            ax.text(
                x,
                y,
                f"{count}\n{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_xticks(np.arange(len(association_order)))
    ax.set_xticklabels(association_order, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(effect_order)))
    ax.set_yticklabels(effect_order)

    ax.set_xlabel("HIV association category")
    ax.set_ylabel("Reported effect direction")

    ax.set_xlim(-0.5, len(association_order) - 0.5)
    ax.set_ylim(-0.5, len(effect_order) - 0.5)

    fig.tight_layout()
    fig.savefig(EFFECT_ASSOCIATION_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return joint


def make_factor_frequency_analysis(non_null, total_articles):
    grouped = (
        non_null
        .groupby("normalised_genetic_factor")
        .agg(
            unique_papers=("paper_id", "nunique"),
            extracted_mentions=("paper_id", "size"),
        )
        .reset_index()
        .sort_values(
            ["unique_papers", "extracted_mentions", "normalised_genetic_factor"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    grouped.insert(0, "Rank", np.arange(1, len(grouped) + 1))

    grouped["Percentage of articles (%)"] = (
        grouped["unique_papers"] / total_articles * 100
    ).round(1)

    top_20 = grouped.head(20).copy()
    top_20.to_csv(TOP_20_CSV, index=False)

    top_10 = grouped.head(10).copy()
    top_10.to_csv(TOP_10_CSV, index=False)

    plot_data = top_20.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 7.5))

    y = np.arange(len(plot_data))

    bars = ax.barh(
        y,
        plot_data["unique_papers"],
        label="Unique papers",
    )

    ax.scatter(
        plot_data["extracted_mentions"],
        y,
        label="Extracted mentions",
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        plot_data["normalised_genetic_factor"]
    )

    ax.set_xlabel("Count")
    ax.set_ylabel("Normalised host genetic factor")
    ax.legend()

    max_value = max(
        plot_data["extracted_mentions"].max(),
        plot_data["unique_papers"].max(),
        1,
    )

    ax.set_xlim(0, max_value * 1.18)

    for bar, paper_count, article_pct in zip(
        bars,
        plot_data["unique_papers"],
        plot_data["Percentage of articles (%)"],
    ):
        ax.text(
            bar.get_width() + max_value * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(paper_count)} ({article_pct:.1f}%)",
            va="center",
            fontsize=8,
        )

    for mention_count, y_pos in zip(
        plot_data["extracted_mentions"],
        y,
    ):
        ax.annotate(
            str(int(mention_count)),
            xy=(mention_count, y_pos),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(TOP_20_FACTORS_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return grouped


def main():
    df = pd.read_csv(
        INPUT_CSV,
        keep_default_na=False,
    )

    required_columns = {
        "paper_id",
        "genetic_factor",
        "normalised_genetic_factor",
        "factor_type",
        "association",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    non_null = df[
        df["genetic_factor"].apply(is_non_null_factor)
    ].copy()

    non_null["effect_direction"] = non_null[
        "factor_type"
    ].apply(normalise_effect)

    non_null["association_category"] = non_null[
        "association"
    ].apply(normalise_association)

    save_summary_table(
        df,
        non_null,
    )

    make_records_per_article_figure(
        df,
        non_null,
    )

    non_null, entity_counts = make_entity_analysis(
        non_null
    )

    make_distribution_table(
        non_null,
        entity_counts,
    )

    make_effect_association_figure(
        non_null
    )

    make_factor_frequency_analysis(
        non_null,
        total_articles=df["paper_id"].nunique(),
    )

    print("\nLarge-scale analysis complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()