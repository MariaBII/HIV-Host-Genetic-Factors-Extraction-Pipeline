import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET

# Dataset
# The same full-text retrieval procedure was used for the 20-paper pilot,
# 100-paper evaluation, and 1200-paper large-scale extraction datasets.

DATASET = "europe_pmc_articles_100"

INPUT_CSV = f"{DATASET}.csv"
OUTPUT_FOLDER = "full_text_100"
OUTPUT_CSV = f"{DATASET}_metadata.csv"

# Create output folder if it does not already exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Download full-text XML from Europe PMC
def download_xml(pmcid):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

    try:
        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            return response.text

        return None

    except requests.RequestException:
        return None

# Extract complete text from an XML element
def clean_element_text(element):
    return " ".join("".join(element.itertext()).split()).strip()

# Find an XML element by tag name
def find_element(root, tag_name):
    for element in root.iter():
        if element.tag.split("}")[-1] == tag_name:
            return element

    return None

# Convert article XML to plain text
def extract_full_text(xml_content):
    try:
        root = ET.fromstring(xml_content)
        parts = []

        # Abstract
        abstract = find_element(root, "abstract")

        if abstract is not None:
            parts.append("Abstract")

            for element in abstract.iter():
                if element.tag.split("}")[-1] == "p":
                    text = clean_element_text(element)

                    if text:
                        parts.append(text)

        # Main article body
        body = find_element(root, "body")

        if body is not None:
            for element in body.iter():
                if element.tag.split("}")[-1] == "p":
                    text = clean_element_text(element)

                    if text:
                        parts.append(text)

        return "\n\n".join(parts)

    except ET.ParseError:
        return None

# Load selected article dataset
df = pd.read_csv(INPUT_CSV)

full_text_files = []

# Download and save full-text articles
for _, row in df.iterrows():

    pmcid = row["paper_id"]

    if pd.isna(pmcid) or not str(pmcid).strip():
        full_text_files.append("")
        continue

    pmcid = str(pmcid).strip()

    print(f"Downloading {pmcid}...")

    xml = download_xml(pmcid)

    if xml is None:
        print(f"Failed to download {pmcid}")
        full_text_files.append("")
        continue

    text = extract_full_text(xml)

    if not text:
        print(f"Failed to extract text from {pmcid}")
        full_text_files.append("")
        continue

    filename = f"{pmcid}.txt"
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    full_text_files.append(
        os.path.join(OUTPUT_FOLDER, filename).replace("\\", "/")
    )


# Add full-text file locations to metadata
df["full_text_file"] = full_text_files

if "paper_type" not in df.columns:
    df["paper_type"] = ""


columns = [
    "paper_id",
    "pmid",
    "title",
    "authors",
    "publication_year",
    "paper_type",
    "journal",
    "europe_pmc_link",
    "full_text_url",
    "abstract",
    "full_text_file",
]

# Keep only columns that exist
columns = [column for column in columns if column in df.columns]

df = df[columns]

# Save updated metadata
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print("\nDone.")