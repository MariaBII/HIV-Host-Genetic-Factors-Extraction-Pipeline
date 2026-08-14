import pandas as pd

INPUT_FILE = "europe_pmc_articles.csv"
OUTPUT_FILE = "screened_pmc_articles.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Remove duplicates
initial_records = len(df)

# Remove duplicate PMIDs
df = df.drop_duplicates(subset="pmid")

# Remove duplicate PMCIDs (paper_id)
df = df.drop_duplicates(subset="paper_id")
duplicates_removed = initial_records - len(df)

# Remove records without PMCID
initial_records = len(df)

df = df[df["paper_id"].notna()]
df = df[df["paper_id"].astype(str).str.strip() != ""]

removed = initial_records - len(df)

# Clean text field
text_columns = [
    "title",
    "authors",
    "journal",
    "abstract"
]

for column in text_columns:
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

# Clean links
link_columns = ["full_text_url", "europe_pmc_link"]

for column in link_columns:
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

df = df[
    [
        "paper_id",
        "pmid",
        "title",
        "authors",
        "publication_year",
        "journal",
        "abstract",
        "full_text_url",
        "europe_pmc_link"
    ]
]

# Save dataset
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"Dataset saved as: {OUTPUT_FILE}")