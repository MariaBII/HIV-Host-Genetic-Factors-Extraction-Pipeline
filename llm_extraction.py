import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# GPT-5 pricing per 1K tokens
MODEL_PRICING = {
    "gpt-5": {
        "input": 0.00125,   # USD per 1K tokens
        "output": 0.01      # USD per 1K tokens
    }
}

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
# Configuration shown for the 100-paper GPT-5 full-text evaluation.
# Other OpenAI models can be used by changing the model setting.
# Claude models require the corresponding Anthropic API implementation.

DATASET = "europe_pmc_articles_100"
MODEL = "gpt-5"

INPUT_CSV = f"{DATASET}_metadata.csv"
OUTPUT_JSON = f"{DATASET}_{MODEL}_extractions.json"
OUTPUT_STATS = f"{DATASET}_{MODEL}_stats.csv"
FAILED_PAPERS = f"{DATASET}_{MODEL}_failed.csv"

# Load metadata
df = pd.read_csv(INPUT_CSV)

def generate_direct_extraction_prompt(full_text, paper_id):
    return f"""
You are a biomedical information extraction expert.
Read the FULL TEXT of a scientific article and extract all **explicitly stated human host genetic factors** 
that have a documented association with **HIV susceptibility, progression, resistance, or treatment response**.

---------------------
### 1. Extraction Principles
Extract only facts that are **supported by evidence in the text**.
Do not invent or assume information that is not stated.

A valid extraction must meet **all** these conditions:
1. The factor is **human genomic** — a gene, allele, SNP, polymorphism, genotype, or haplotype (e.g., CCR5-Δ32, HLA-B*57:01, rs2395029).
2. It is **explicitly linked to an HIV-related host outcome** (susceptibility, progression, resistance, or treatment response) 
   in the same sentence, paragraph, or study section.
3. The association is **supported by textual evidence** (e.g., “was associated with”, “was linked to”, “conferred protection”, “increased risk”).

If any of these conditions are missing → do not extract.

---------------------
### 2. Exclude
Do NOT extract:
- Viral mutations or polymorphisms (e.g., K103N, M41L, L74M)
- Drug resistance mutations in viral genes (NNRTI, NRTI, PI, INSTI)
- Non-genetic immune or clinical markers (e.g., cytokines, CD4 count, viral load)
- Treatment regimens, drugs, or pharmacological effects not linked to host DNA
- Demographic, behavioural, or environmental factors (e.g., age, sex, occupation, transmission route)
- General hypotheses, background context, or speculative discussion without evidence

---------------------
### 3. Field Definitions
For each valid extraction, include:

- **genetic_factor** → the human gene, allele, or variant (e.g., "CCR5-Δ32", "HLA-B*57:01", "rs2395029")
- **population** → extract any explicitly stated study population (ethnicity, ancestry, country, or region).  
  If population is mentioned anywhere in the same paragraph or study description and can be plausibly linked to the genetic finding, include it.  
  Otherwise, return "None".
- **factor_type**
  - "Protective" → associated with reduced HIV susceptibility, slower progression, or resistance
  - "Risk" → associated with increased HIV susceptibility or faster progression
  - "Mixed" → both effects reported in the same article
  - "Unclear" → direction of effect not specified
- **association**
  - "Susceptibility" → linked to risk of HIV infection/acquisition
  - "Progression" → affects rate or severity of disease progression
  - "Resistance" → confers protection against HIV infection or disease
  - "Treatment Response" → influences ART efficacy or adverse reactions
  - "Unclear" → insufficient detail for classification
- **quote** → the exact sentence(s) from the text that directly support the association. Use only verbatim text — no paraphrasing.

---------------------
### 4. Output Format
Return a **strictly valid JSON array**, where each item follows this schema:

- "paper_id": "{paper_id}"
- "genetic_factor": string|null
- "population": string|null
- "factor_type": "Protective" | "Risk" | "Mixed" | "Unclear" | null
- "association": "Susceptibility" | "Progression" | "Resistance" | "Treatment Response" | "Unclear" | null
- "quote": string

If no valid human genetic factors are found, return exactly:

[
  {{
    "paper_id": "{paper_id}",
    "genetic_factor": null,
    "population": null,
    "factor_type": null,
    "association": null,
    "quote": "No human genetic factors associated with HIV found in the full text."
  }}
]

---------------------
### 5. Output Rules
- Output **only** the JSON array — no explanations, no commentary.
- Each extracted fact must be supported by explicit evidence in the text.
- If uncertain, include the finding **only if it is plausibly supported** by the article, but never invent or assume information.

---------------------
FULL TEXT:
{full_text}
"""

# GPT call
def chat_with_gpt(prompt, model=MODEL):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        content = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)
        return content, input_tokens, output_tokens
    except Exception as e:

        print(f"API request failed: {e}")

        return "", 0, 0

# initialise
json_results = []
failed_papers = []
extraction_stats = []

total_input_tokens = 0
total_output_tokens = 0

llm_model = MODEL
# process papers
for _, row in df.iterrows():
    paper_id = row["paper_id"]
    file_path = row["full_text_file"]

    if not os.path.exists(file_path):
        print(f"Full text file not found: {file_path}")
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read().strip()

    if not full_text:
        print(f"⚠️ Empty full text file for {paper_id}")
        continue

    print(f"Processing paper {paper_id}...")

    prompt = generate_direct_extraction_prompt(full_text, paper_id)
    extraction, input_tokens, output_tokens = chat_with_gpt(prompt)
    time.sleep(1)

    if extraction == "":

        failed_papers.append({
            "paper_id": paper_id,
            "reason": "API request failed"
        })

        continue

    input_cost = (
        input_tokens / 1000
    ) * MODEL_PRICING[MODEL]["input"]

    output_cost = (
        output_tokens / 1000
    ) * MODEL_PRICING[MODEL]["output"]

    total_cost = input_cost + output_cost

    extraction_stats.append({

        "paper_id": paper_id,
        "model": MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "extraction_type": "full_text"

    })

    json_results.append(extraction)

    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

# Save stats to CSV
stats_df = pd.DataFrame(extraction_stats)
stats_df.to_csv(OUTPUT_STATS, index=False)

# Save JSON output
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    for r in json_results:
        f.write(r + "\n\n")

# save failed papers
failed_df = pd.DataFrame(failed_papers)
failed_df.to_csv(FAILED_PAPERS, index=False)

# Calculate total cost
input_rate = MODEL_PRICING[MODEL]["input"]
output_rate = MODEL_PRICING[MODEL]["output"]

estimated_total_cost = ((total_input_tokens / 1000) * input_rate + (total_output_tokens / 1000) * output_rate)

print("\nDone.")
print(f"Processed: {len(extraction_stats)}")
print(f"Failed: {len(failed_papers)}")
print(f"Estimated cost: ${estimated_total_cost:.4f}")