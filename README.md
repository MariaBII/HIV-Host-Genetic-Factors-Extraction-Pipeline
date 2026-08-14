# HIV Host Genetic Factors Extraction Pipeline

## Description

This repository contains the Python pipeline developed for the MSc Bioinformatics project to extract human host genetic factors associated with HIV from biomedical literature using large language models (LLMs) accessed via API. 

The project used Europe PMC full-text articles, manually verified gold-standard datasets, and multiple OpenAI and Anthropic models for extraction and evaluation. 

---
## Running the Pipeline

   **1. Set up and activate a virtual environment (optional)**
   - Create and activate a virtual environment if you prefer to keep the project dependencies isolated.

   ```bash
   python -m venv .venv
   ```

   On Windows:

   ```bash
   .venv\Scripts\activate
   ```

   **2. Install the required packages**
   - Install the necessary Python packages using:

   ```bash
   python -m pip install -r requirements.txt
   ```

   **3. Configure the OpenAI API key**
   - Create a local `.env` file containing:

   ```text
   OPENAI_API_KEY=your_api_key_here
   ```

   Do not upload the `.env` file or API key to the public repository.

   **4. Run the scripts**
   - Run the required script from the command line, for example:

   ```bash
   python retrieve_epmc_articles.py
   ```

   The main scripts should be run in the order described in the workflow below.

## Python Scripts

**1. Retrieve Europe PMC articles**
   - Run `retrieve_epmc_articles.py` to query the Europe PMC REST API for open-access HIV publications.
   - Retrieves PMCID, PMID, title, authors, publication year, journal, abstract and Europe PMC links.
   - Produces `europe_pmc_articles.csv`.

   ```bash
   python retrieve_epmc_articles.py
   ```

**2. Screen Europe PMC metadata**
   - Run `screening_epmc_articles.py` to remove duplicates, exclude entries without a PMCID and clean the metadata.
   - Produces `screened_pmc_articles.csv`.

   ```bash
   python screening_epmc_articles.py
   ```

**3. Download and process full-text articles**
   - Run `download_full_text.py` to download open-access full-text XML from Europe PMC.
   - Also creates an updated metadata CSV containing the location of each full-text file.
   - The current configuration is for the 100-paper evaluation dataset.

   ```bash
   python download_full_text.py
   ```

**4. Extract host genetic factors**
   - Run `llm_extraction.py` to process full-text articles using the final extraction prompt and the OpenAI API.
   - Extracts the genetic factor, population, factor type, HIV association category and supporting quotation.
   - Produces JSON extraction output, token and cost statistics, and a file containing failed requests.
   - The current configuration is for the 100-paper GPT-5 full-text evaluation.

   ```bash
   python llm_extraction.py
   ```

**5. Convert JSON output to CSV**
   - Run `json_to_csv.py` to convert the LLM JSON output into a structured CSV file for evaluation and analysis.
   - Retains the required extraction fields and adds the model name and input type.

   ```bash
   python json_to_csv.py
   ```

**6. Evaluate LLM performance**
   - Run `evaluate_llm_performance.py` to compare abstract and full-text LLM outputs against the gold standard dataset.
   - Uses fuzzy-matching thresholds of 70 for genetic factors and 60 for factor type, association and population.
   - Calculates TP, FP, FN, precision, recall and F1-score overall and for each extraction field.
   - Also generates performance summaries and evaluation figures.
   - Requires `gold_dataset_100.csv` and the individual LLM outputs CSV files.
     
   ```bash
   python evaluate_llm_performance.py
   ```

**7. Normalise large-scale genetic factors**
   - Run `normalise_large_scale_factors.py` to standardise genetic factor names from the large scale extraction.
   - Applies conservative normalisation without adding unsupported biological specificity.
   - Produces the normalised dataset and a mapping of raw to normalised factor names.

   ```bash
   python normalise_large_scale_factors.py
   ```

8. **Analyse large-scale extraction results**
   - Run `analyse_large_scale_results.py` to analyse the normalised LLM results from full-text articles.
   - Summarises extraction records, biological entity classes, effect directions, HIV association categories.
   - Saves summary CSV files and figures in `large_scale_analysis/`.

   ```bash
   python analyse_large_scale_results.py
   ```

## Main Data Files

The repository includes the main datasets used in the analysis:

- `europe_pmc_articles_100.csv` – metadata for the 100-paper evaluation dataset.
- `gold_dataset_100.csv` – manually verified 100-paper gold standard dataset used for LLM evaluation.
- `extracted_factors_1200_fulltext_GPT5.csv` – GPT-5 extraction results from the 1,200-paper large-scale analysis.

Additional intermediate and analysis output files can be regenerated using the provided Python scripts.


## Reproducibility

The repository contains the main scripts used to retrieve publications, process full-text articles, perform LLM extraction, evaluate model performance, normalise extraction results and generate the analyses reported in the dissertation.

The pipeline can be adapted to other article datasets by changing the input files and configuration settings. 

The extraction script can also be configured for other OpenAI models and other LLMs, such as Anthropic Claude, using the corresponding API implementation.

LLM outputs may vary depending on model availability, model version and API configuration.

---

## Author

**Maria Bolea**

MSc Bioinformatics  
Queen Mary University of London  
2026
