"""
SEC Form 8-K Filing Collector - S&P 500
UCF Research Project

This version DOES NOT run a local LLM or Ollama.
It collects recent 8-K filing text from a broad range of S&P 500
companies and saves small JSONL batches for analysis with ChatGPT.

Outputs:
- sec_8k_filing_batches/batch_001.jsonl, etc.
- sec_8k_filings_collected.csv
- sec_8k_collection_review.csv
"""

import json
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# -------------------- SETTINGS --------------------

SEC_EMAIL = "la146090@ucf.edu"

# Target up to 100 different companies, one recent 8-K per company.
MAX_COMPANIES_WITH_FILINGS = 100

# Keep each filing reasonably small for later ChatGPT batch review.
MAX_FILING_CHARS = 9000

REQUEST_TIMEOUT = 40
SEC_PAUSE = 0.15

BASE_DIR = Path(__file__).resolve().parent
BATCH_DIR = BASE_DIR / "sec_8k_filing_batches"

HEADERS = {
    "User-Agent": f"UCF SEC 8-K Research Project {SEC_EMAIL}",
    "Accept-Encoding": "gzip, deflate",
}
WIKIPEDIA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

def normalize_ticker(ticker):
    return str(ticker).strip().upper().replace(".", "-")

def get_sp500_companies():
    print("Loading S&P 500 companies...", flush=True)
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(
        wiki_url, headers=WIKIPEDIA_HEADERS, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    sp500 = tables[0].rename(
        columns={"Symbol": "ticker", "Security": "company_name"}
    )
    sp500["ticker"] = sp500["ticker"].astype(str).str.strip()
    sp500["normalized_ticker"] = sp500["ticker"].apply(normalize_ticker)

    sec_url = "https://www.sec.gov/files/company_tickers.json"
    sec_response = requests.get(
        sec_url, headers=HEADERS, timeout=REQUEST_TIMEOUT
    )
    sec_response.raise_for_status()
    sec_data = sec_response.json()

    sec_rows = []
    for item in sec_data.values():
        sec_rows.append({
            "normalized_ticker": normalize_ticker(item["ticker"]),
            "cik": int(item["cik_str"]),
        })
    sec_df = pd.DataFrame(sec_rows)

    companies = sp500.merge(sec_df, on="normalized_ticker", how="inner")
    companies = companies.drop_duplicates("normalized_ticker")
    companies = companies[["company_name", "ticker", "cik"]].copy()
    print(f"Companies matched to SEC: {len(companies)}", flush=True)
    return companies

def get_latest_8k(session, company):
    cik = str(int(company["cik"])).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    recent = response.json().get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    docs = recent.get("primaryDocument", [])

    for form, accession, filing_date, document in zip(
        forms, accessions, dates, docs
    ):
        if form == "8-K":
            accession_clean = accession.replace("-", "")
            filing_url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{int(company['cik'])}/{accession_clean}/{document}"
            )
            return {
                "company_name": company["company_name"],
                "stock_name": company["ticker"],
                "cik": int(company["cik"]),
                "filing_time": filing_date,
                "accession_number": accession,
                "filing_url": filing_url,
            }
    return None

def download_filing_text(session, filing_url):
    response = session.get(filing_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    for element in soup(["script", "style", "ix:header", "ix:hidden"]):
        element.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:MAX_FILING_CHARS]

def main():
    if not SEC_EMAIL or "@" not in SEC_EMAIL:
        raise ValueError("Enter a valid contact email in SEC_EMAIL.")

    BATCH_DIR.mkdir(exist_ok=True)
    for old_file in BATCH_DIR.glob("batch_*.jsonl"):
        old_file.unlink()

    session = requests.Session()
    session.headers.update(HEADERS)

    companies = get_sp500_companies()

    print(
        "Checking each matched company for its latest 8-K. "
        "This stage does not use an LLM.",
        flush=True,
    )
    filings = []
    errors = []

    for index, (_, company) in enumerate(companies.iterrows(), start=1):
        try:
            filing = get_latest_8k(session, company)
            if filing:
                filings.append(filing)
        except Exception as exc:
            errors.append({
                "stock_name": company["ticker"],
                "company_name": company["company_name"],
                "error": str(exc),
            })

        if index % 25 == 0:
            print(
                f"Companies checked: {index}/{len(companies)} | "
                f"companies with a recent 8-K: {len(filings)}",
                flush=True,
            )
        time.sleep(SEC_PAUSE)

    if not filings:
        raise RuntimeError("No recent 8-K filings were found.")

    # Choose the most recent filings across distinct companies.
    filings_df = pd.DataFrame(filings)
    filings_df["filing_time_sort"] = pd.to_datetime(
        filings_df["filing_time"], errors="coerce"
    )
    filings_df = filings_df.sort_values(
        "filing_time_sort", ascending=False
    ).head(MAX_COMPANIES_WITH_FILINGS)
    filings_df = filings_df.drop(columns=["filing_time_sort"])
    selected = filings_df.to_dict("records")

    print(
        f"\nSelected {len(selected)} filings from "
        f"{filings_df['stock_name'].nunique()} different companies.",
        flush=True,
    )

    # Download text and write five filings per batch.
    batch_size = 5
    review_rows = []
    collected_rows = []
    batch_number = 1
    batch_records = []

    for index, filing in enumerate(selected, start=1):
        print(
            f"Downloading filing {index}/{len(selected)}: "
            f"{filing['stock_name']} ({filing['filing_time']})",
            flush=True,
        )
        try:
            filing_text = download_filing_text(
                session, filing["filing_url"]
            )
            if len(filing_text) < 200:
                raise ValueError("Extracted filing text was too short.")

            record = {
                "company_name": filing["company_name"],
                "stock_name": filing["stock_name"],
                "filing_time": filing["filing_time"],
                "accession_number": filing["accession_number"],
                "filing_url": filing["filing_url"],
                "filing_text": filing_text,
            }
            batch_records.append(record)
            collected_rows.append({
                key: filing[key] for key in [
                    "company_name", "stock_name", "filing_time",
                    "accession_number", "filing_url"
                ]
            })
            status = "Text collected"
            error_message = ""
        except Exception as exc:
            status = "Download error"
            error_message = str(exc)
            errors.append({
                "stock_name": filing["stock_name"],
                "company_name": filing["company_name"],
                "error": error_message,
            })

        review_rows.append({
            "company_name": filing["company_name"],
            "stock_name": filing["stock_name"],
            "filing_time": filing["filing_time"],
            "accession_number": filing["accession_number"],
            "filing_url": filing["filing_url"],
            "status": status,
            "error": error_message,
        })

        if len(batch_records) == batch_size or index == len(selected):
            if batch_records:
                batch_path = BATCH_DIR / f"batch_{batch_number:03d}.jsonl"
                with open(batch_path, "w", encoding="utf-8") as out:
                    for item in batch_records:
                        out.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(
                    f"Saved {batch_path.name} "
                    f"({len(batch_records)} filings)",
                    flush=True,
                )
                batch_number += 1
                batch_records = []

        pd.DataFrame(collected_rows).to_csv(
            BASE_DIR / "sec_8k_filings_collected.csv", index=False
        )
        pd.DataFrame(review_rows).to_csv(
            BASE_DIR / "sec_8k_collection_review.csv", index=False
        )
        time.sleep(SEC_PAUSE)

    if errors:
        pd.DataFrame(errors).to_csv(
            BASE_DIR / "sec_8k_collection_errors.csv", index=False
        )

    print("\nCOLLECTION COMPLETE", flush=True)
    print(f"Filings selected: {len(selected)}", flush=True)
    print(f"Filing texts collected: {len(collected_rows)}", flush=True)
    print(
        "Companies represented: "
        f"{pd.DataFrame(collected_rows)['stock_name'].nunique() if collected_rows else 0}",
        flush=True,
    )
    print(f"Batch folder: {BATCH_DIR}", flush=True)
    print(
        "Next: upload batch_001.jsonl here for LLM-based analysis.",
        flush=True,
    )

if __name__ == "__main__":
    main()
