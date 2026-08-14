import re
import unicodedata
import pandas as pd

# Large-scale GPT-5 extraction output
INPUT_CSV = "extracted_factors_1200_fulltext_GPT5.csv"
OUTPUT_CSV = "extracted_factors_1200_fulltext_GPT5_normalised.csv"
MAPPING_CSV = "large_scale_factor_normalisation_mapping.csv"


NONE_LIKE = {"", "none", "null", "nan", "na"}


def raw_factor_text(value):
    """Return the original factor string with only surrounding whitespace removed."""
    if value is None:
        return ""
    return str(value).strip()


def is_none_like(value):
    return raw_factor_text(value).lower() in NONE_LIKE


def clean_factor_text(value):
    """Light text cleaning used only before factor-name normalisation."""
    text = raw_factor_text(value)

    if is_none_like(text):
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = (
        text.replace("∗", "*")
        .replace("‐", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalise_ccr5_delta32(text):
    # Harmonise clear CCR5 Δ32 notation variants without merging generic CCR5.
    pattern = re.compile(
        r"\bCCR5\b[\s\-_/]*(?:Δ|δ|delta|del(?:etion)?)?[\s\-_/]*32"
        r"(?:\s*bp)?(?:\s*\(rs333\))?",
        flags=re.IGNORECASE,
    )

    if pattern.search(text):
        return "CCR5Δ32"

    reverse_pattern = re.compile(
        r"(?:Δ|δ|delta)[\s\-_/]*32(?:\s*bp)?"
        r".{0,25}\bCCR5\b",
        flags=re.IGNORECASE,
    )

    if reverse_pattern.search(text):
        return "CCR5Δ32"

    return None


def normalise_hla(text):
    value = text.replace("∗", "*")

    # Standardise HLA spacing.
    value = re.sub(
        r"\bHLA\s*-\s*",
        "HLA-",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\bHLA\s+([A-Z0-9]+)",
        r"HLA-\1",
        value,
        flags=re.IGNORECASE,
    )

    # HLA-B5701 / HLA-B*5701 -> HLA-B*57:01
    value = re.sub(
        r"\bHLA-([A-Z0-9]+)\s*\*?\s*(\d{2})(\d{2})\b",
        r"HLA-\1*\2:\3",
        value,
        flags=re.IGNORECASE,
    )

    # HLA-B57 / HLA-B*57 -> HLA-B*57
    value = re.sub(
        r"\bHLA-([A-Z0-9]+)\s*\*?\s*(\d{2})\b",
        r"HLA-\1*\2",
        value,
        flags=re.IGNORECASE,
    )

    # B*5701 -> HLA-B*57:01
    value = re.sub(
        r"(?<![A-Z0-9-])([ABC])\*(\d{2})(\d{2})\b",
        r"HLA-\1*\2:\3",
        value,
        flags=re.IGNORECASE,
    )

    # B*57 -> HLA-B*57
    value = re.sub(
        r"(?<![A-Z0-9-])([ABC])\*(\d{2})\b",
        r"HLA-\1*\2",
        value,
        flags=re.IGNORECASE,
    )

    # Standardise high-resolution allele spacing/case.
    value = re.sub(
        r"\bHLA-([A-Z0-9]+)\s*\*\s*(\d{2,3})\s*:\s*(\d{2,3})\b",
        lambda m: f"HLA-{m.group(1).upper()}*{m.group(2)}:{m.group(3)}",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\bHLA-([A-Z0-9]+)\s*\*\s*(\d{2,3})\b",
        lambda m: f"HLA-{m.group(1).upper()}*{m.group(2)}",
        value,
        flags=re.IGNORECASE,
    )

    # Remove generic descriptor when the value is otherwise one allele.
    value = re.sub(
        r"^(HLA-[A-Z0-9]+\*\d{2,3}(?::\d{2,3})?)"
        r"\s+(?:allele|genotype|variant)$",
        r"\1",
        value,
        flags=re.IGNORECASE,
    )

    return value


def normalise_rsid(text):
    # Standardise rs identifier case and spacing.
    text = re.sub(
        r"\bRS\s*(\d+)\b",
        lambda m: f"rs{m.group(1)}",
        text,
        flags=re.IGNORECASE,
    )

    # rs123-A-G / rs123:A/G -> rs123 A>G
    text = re.sub(
        r"\b(rs\d+)\s*[-:]\s*([ACGT])\s*[-/]\s*([ACGT])\b",
        lambda m: (
            f"{m.group(1)} "
            f"{m.group(2).upper()}>{m.group(3).upper()}"
        ),
        text,
        flags=re.IGNORECASE,
    )

    # rs123 (A/G) -> rs123 A>G
    text = re.sub(
        r"\b(rs\d+)\s*\(\s*([ACGT])\s*[>/]\s*([ACGT])\s*\)",
        lambda m: (
            f"{m.group(1)} "
            f"{m.group(2).upper()}>{m.group(3).upper()}"
        ),
        text,
        flags=re.IGNORECASE,
    )

    return text


def normalise_known_names(text):
    replacements = [
        (r"\bTRIM5\s*[αa]\b", "TRIM5"),
        (r"\bPON-?1\b", "PON1"),
        (r"\bPON-?2\b", "PON2"),
        (r"\bAPOBEC\s*3G\b", "APOBEC3G"),
        (r"\bSAMHD\s*1\b", "SAMHD1"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return text


def normalise_factor(value):
    # Preserve null extraction rows explicitly.
    if is_none_like(value):
        return "None"

    text = clean_factor_text(value)

    ccr5 = normalise_ccr5_delta32(text)

    if ccr5:
        return ccr5

    text = normalise_hla(text)
    text = normalise_rsid(text)
    text = normalise_known_names(text)

    text = text.strip(" ,;.")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    # keep_default_na=False preserves literal "None" values from the input CSV.
    df = pd.read_csv(
        INPUT_CSV,
        keep_default_na=False,
    )

    if "genetic_factor" in df.columns:
        factor_column = "genetic_factor"
    elif "host_genetic_factor" in df.columns:
        factor_column = "host_genetic_factor"
    else:
        raise ValueError(
            "No genetic_factor or host_genetic_factor column found."
        )

    raw_values = df[factor_column].apply(raw_factor_text)

    normalised_values = df[factor_column].apply(
        normalise_factor
    )

    insert_at = df.columns.get_loc(factor_column) + 1

    df.insert(
        insert_at,
        "normalised_genetic_factor",
        normalised_values,
    )

    # Save all rows, including the null extraction records.
    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    # Build mapping using only non-null factor records.
    mapping_source = pd.DataFrame(
        {
            "raw_genetic_factor": raw_values,
            "normalised_genetic_factor": normalised_values,
        }
    )

    mapping_source = mapping_source[
        ~mapping_source["raw_genetic_factor"]
        .str.lower()
        .isin(NONE_LIKE)
    ].copy()

    mapping = (
        mapping_source
        .value_counts(
            [
                "raw_genetic_factor",
                "normalised_genetic_factor",
            ]
        )
        .reset_index(name="records")
        .sort_values(
            [
                "normalised_genetic_factor",
                "records",
            ],
            ascending=[True, False],
        )
    )

    mapping.to_csv(
        MAPPING_CSV,
        index=False,
        encoding="utf-8",
    )

    # Non-null records only for factor-level analysis.
    non_null_mask = ~raw_values.str.lower().isin(
        NONE_LIKE
    )

    non_null = df[non_null_mask].copy()

    # Raw distinct count uses the original strings, without notation cleaning.
    raw_unique = raw_values[
        non_null_mask
    ].nunique()

    normalised_unique = non_null[
        "normalised_genetic_factor"
    ].nunique()

    null_records = len(df) - len(non_null)

    print("\nNormalisation complete.")
    print(f"Input rows: {len(df)}")
    print(
        f"Non-null extraction records: "
        f"{len(non_null)}"
    )
    print(
        f"Null extraction records preserved: "
        f"{null_records}"
    )
    print(
        f"Distinct raw factor strings: "
        f"{raw_unique}"
    )
    print(
        f"Distinct normalised factors: "
        f"{normalised_unique}"
    )
    print(
        f"Normalised CSV saved to: "
        f"{OUTPUT_CSV}"
    )
    print(
        f"Mapping saved to: "
        f"{MAPPING_CSV}"
    )

    if "paper_id" in non_null.columns:
        mentions = (
            non_null
            .groupby("normalised_genetic_factor")
            .size()
            .rename("mentions")
        )

        unique_papers = (
            non_null
            .groupby("normalised_genetic_factor")[
                "paper_id"
            ]
            .nunique()
            .rename("unique_papers")
        )

        top = (
            pd.concat(
                [unique_papers, mentions],
                axis=1,
            )
            .sort_values(
                [
                    "unique_papers",
                    "mentions",
                ],
                ascending=False,
            )
            .head(10)
        )

        print("\nTop 10 normalised factors:")
        print(top.to_string())


if __name__ == "__main__":
    main()