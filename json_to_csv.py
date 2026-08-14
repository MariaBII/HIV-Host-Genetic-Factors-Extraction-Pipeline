import json
import pandas as pd

# JSON to CSV conversion is kept separate from LLM extraction so that
# parsing can be rerun without repeating API calls if conversion fails.

# # Configuration shown for the 100-paper GPT-5 full-text evaluation.
# The same conversion procedure can be used for other datasets and model outputs.
DATASET = "europe_pmc_articles_100"
MODEL = "gpt-5"

INPUT_JSON = f"{DATASET}_{MODEL}_extractions.json"
OUTPUT_CSV = f"{DATASET}_{MODEL}_extractions.csv"

# Load json
parsed_data = []

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    lines = f.readlines()

block_lines = []
for line in lines:
    if line.strip() == "" and block_lines:
        block = "".join(block_lines).strip()
        try:
            data = json.loads(block)
            if isinstance(data, list):
                parsed_data.extend(data)
            elif isinstance(data, dict):
                parsed_data.append(data)
        except Exception as e:
            print(f"Skipping invalid JSON block: {e}")
        block_lines = []
    else:
        block_lines.append(line)

# Handle final json block
if block_lines:
    block = "".join(block_lines).strip()
    try:
        data = json.loads(block)
        if isinstance(data, list):
            parsed_data.extend(data)
        elif isinstance(data, dict):
            parsed_data.append(data)
    except Exception as e:
        print(f"Skipping invalid JSON block: {e}")

# Create dataframe
df = pd.DataFrame(parsed_data)

# Add metadata 
df["model"] = MODEL
df["extraction_type"] = "full_text"

# Ensure expected columns exist
expected_columns = [
    "paper_id",
    "genetic_factor",
    "population",
    "factor_type",
    "association",
    "quote",
    "model",
    "extraction_type"
]

for column in expected_columns:
    if column not in df.columns:
        df[column] = None
df = df[expected_columns]

# Replace missing values
df.fillna("None", inplace=True)

# Save CSV
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

# Summary
print("\nDone.")
print(f"Rows extracted: {len(df)}")
print(f"CSV saved to: {OUTPUT_CSV}")