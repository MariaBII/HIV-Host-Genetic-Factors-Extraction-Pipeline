import time
import requests
import pandas as pd

# Configuration
BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

QUERY = """ (HIV OR HIV-1) AND (SNP OR mutation OR polymorphism OR gene OR HLA OR genetic*) AND OPEN_ACCESS:y"""

PAGE_SIZE = 100
MAX_RESULTS = 10000

# Retrieve articles from Europe PMC
def retrieve_articles():
    articles = []
    cursor_mark = "*"
    while len(articles) < MAX_RESULTS:
        params = {
            "query": QUERY,
            "format": "json",
            "resultType": "core",
            "pageSize": PAGE_SIZE,
            "cursorMark": cursor_mark
        }
        response = requests.get(BASE_URL, params=params, timeout=30)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
        results = response.json()["resultList"]["result"]
        if not results:
            break

        # Store metadata for each article
        for article in results:

            articles.append({
                "paper_id": article.get("pmcid", ""),
                "pmid": article.get("pmid", ""),
                "title": article.get("title", ""),
                "authors": article.get("authorString", ""),
                "publication_year": article.get("pubYear", ""),
                "journal": article.get("journalTitle", ""),
                "abstract": article.get("abstractText", ""),

                "full_text_url":
                    f"https://europepmc.org/articles/{article.get('pmcid','')}"
                    if article.get("pmcid") else "",

                "europe_pmc_link":
                    f"https://europepmc.org/article/MED/{article.get('pmid','')}"
                    if article.get("pmid") else ""

            })

            if len(articles) >= MAX_RESULTS:
                break

        # Retrive the next page of results   
        cursor_mark = response.json().get("nextCursorMark")
        print(f"Retrieved {len(articles)} papers")
        # Europe PMC API time
        time.sleep(1)

    return pd.DataFrame(articles)

def main():
    df = retrieve_articles()
    df.to_csv( "europe_pmc_articles.csv", index=False, encoding="utf-8")
    
    print(f"\nFinished: {len(df)} papers saved.")

if __name__ == "__main__":
    main()