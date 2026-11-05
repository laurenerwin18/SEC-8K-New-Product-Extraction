
"""
SEC Form 8-K New Product Extraction - S&P 500
UCF Research Project

Uses:
- S&P 500 constituent list
- SEC EDGAR submissions API
- BeautifulSoup
- Local Ollama LLM

Outputs:
- sec_8k_new_products.csv
- sec_8k_filings_review.csv
- sec_8k_errors.txt
"""

import json
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# 1. SETTINGS
# ============================================================

SEC_EMAIL = "la146090@ucf.edu"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

MAX_FILINGS_TO_ANALYZE = 200
MAX_COMPANIES_TO_SCAN = None

MAX_FILING_CHARS = 18000

REQUEST_TIMEOUT = 45
OLLAMA_TIMEOUT = 300

SEC_PAUSE = 0.15
OLLAMA_PAUSE = 0.2

OUTPUT_CSV = "sec_8k_new_products.csv"
REVIEW_CSV = "sec_8k_filings_review.csv"
ERROR_LOG = "sec_8k_errors.txt"

BASE_DIR = Path(__file__).resolve().parent

# Do not manually set a Host header.
# Requests will set the correct host for each website.
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

PRODUCT_COLUMNS = [
    "company_name",
    "stock_name",
    "filing_time",
    "new_product",
    "product_description",
]

REVIEW_COLUMNS = [
    "company_name",
    "stock_name",
    "filing_time",
    "accession_number",
    "filing_url",
    "status",
    "products_found",
]


# ============================================================
# 2. GENERAL HELPER FUNCTIONS
# ============================================================

def get_session():
    """Create an HTTP session with the SEC contact information."""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def normalize_ticker(ticker):
    """Normalize ticker symbols for matching."""
    return str(ticker).strip().upper().replace(".", "-")


def log_error(message):
    """Append errors to the project error log."""
    with open(
        BASE_DIR / ERROR_LOG,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(message + "\n")


# ============================================================
# 3. LOAD S&P 500 COMPANIES AND SEC CIK NUMBERS
# ============================================================

def get_sp500_companies():
    """
    Download the S&P 500 constituent list from Wikipedia.
    Match tickers to official SEC CIK numbers.
    """

    print(
        "Step 1/5: Loading S&P 500 companies...",
        flush=True
    )

    wikipedia_url = (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )

    # Use a separate request with a browser-style User-Agent.
    # This prevents sending SEC-specific headers to Wikipedia.
    response = requests.get(
        wikipedia_url,
        headers=WIKIPEDIA_HEADERS,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    sp500 = tables[0].rename(
        columns={
            "Symbol": "ticker",
            "Security": "company_name"
        }
    )

    sp500["ticker"] = (
        sp500["ticker"].astype(str).str.strip()
    )

    sp500["normalized_ticker"] = (
        sp500["ticker"].apply(normalize_ticker)
    )

    # Download official SEC ticker mapping.
    sec_url = "https://www.sec.gov/files/company_tickers.json"

    sec_response = requests.get(
        sec_url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )

    sec_response.raise_for_status()

    sec_data = sec_response.json()

    sec_records = []

    for item in sec_data.values():
        sec_records.append({
            "normalized_ticker": normalize_ticker(
                item["ticker"]
            ),
            "cik": int(item["cik_str"]),
            "sec_company_name": item["title"]
        })

    sec_df = pd.DataFrame(sec_records)

    companies = sp500.merge(
        sec_df,
        on="normalized_ticker",
        how="inner"
    )

    companies = companies.drop_duplicates(
        subset=["normalized_ticker"]
    )

    companies = companies[
        ["company_name", "ticker", "cik"]
    ].copy()

    if MAX_COMPANIES_TO_SCAN is not None:
        companies = companies.head(
            MAX_COMPANIES_TO_SCAN
        )

    print(
        f"S&P 500 companies matched to SEC: {len(companies)}",
        flush=True
    )

    if companies.empty:
        raise RuntimeError(
            "No S&P 500 companies matched to SEC ticker data."
        )

    return companies


# ============================================================
# 4. RETRIEVE RECENT 8-K FILINGS
# ============================================================

def get_recent_8k_filings(
    session,
    cik,
    ticker,
    company_name,
    remaining_filings
):
    """
    Retrieve recent 8-K filing metadata from the SEC
    submissions API.
    """

    padded_cik = str(cik).zfill(10)

    submissions_url = (
        f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
    )

    response = session.get(
        submissions_url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    documents = recent.get("primaryDocument", [])

    filings = []

    for form, accession, filing_date, document in zip(
        forms,
        accessions,
        filing_dates,
        documents
    ):

        if form != "8-K":
            continue

        accession_clean = accession.replace("-", "")

        filing_url = (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession_clean}/{document}"
        )

        filings.append({
            "company_name": company_name,
            "stock_name": ticker,
            "filing_time": filing_date,
            "accession_number": accession,
            "filing_url": filing_url
        })

        if len(filings) >= remaining_filings:
            break

    return filings


# ============================================================
# 5. DOWNLOAD AND CLEAN SEC FILING TEXT
# ============================================================

def download_filing_text(session, filing_url):
    """Download an SEC filing and extract readable text."""

    response = session.get(
        filing_url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )

    for element in soup([
        "script",
        "style",
        "ix:header",
        "ix:hidden"
    ]):
        element.decompose()

    filing_text = soup.get_text(
        separator=" ",
        strip=True
    )

    filing_text = re.sub(
        r"\s+",
        " ",
        filing_text
    )

    return filing_text[:MAX_FILING_CHARS]


# ============================================================
# 6. CHECK LOCAL OLLAMA CONNECTION
# ============================================================

def test_ollama_connection():
    """Confirm that Ollama is running and the model is installed."""

    print(
        "\nStep 2/5: Checking Ollama connection...",
        flush=True
    )

    response = requests.get(
        "http://localhost:11434/api/tags",
        timeout=10
    )

    response.raise_for_status()

    models = [
        item["name"]
        for item in response.json().get("models", [])
    ]

    model_available = any(
        model == OLLAMA_MODEL
        or model.startswith(OLLAMA_MODEL + ":")
        for model in models
    )

    if not model_available:
        raise RuntimeError(
            f"Model {OLLAMA_MODEL} is not installed. "
            f"Available models: {models}. "
            f"Run: ollama pull {OLLAMA_MODEL}"
        )

    print(
        f"Ollama connected. Model: {OLLAMA_MODEL}",
        flush=True
    )


# ============================================================
# 7. LLM PRODUCT EXTRACTION
# ============================================================

def extract_products_with_llm(
    filing_text,
    company_name
):
    """
    Use Ollama to identify genuinely new products or services
    announced in an SEC 8-K filing.
    """

    prompt = f"""
You are analyzing an SEC Form 8-K filing.

Company: {company_name}

Your task is to identify genuinely NEW products or services
that the company announces, launches, introduces, or releases.

STRICT RULES:

1. Only identify products or services that are genuinely new.

2. Do not classify earnings, financial results, debt,
   bonds, stock, securities, dividends, acquisitions,
   investor presentations, or financial statements as products.

3. Do not classify an existing product as new unless the
   filing clearly announces a new launch or release.

4. Do not invent product names or product descriptions.

5. If there are no new products or services, return an
   empty products list.

6. Keep each product description under 180 characters.

7. Return valid JSON only.

Use this exact structure:

{{
    "products": [
        {{
            "new_product": "Product name",
            "product_description": "Short factual description"
        }}
    ]
}}

SEC FILING TEXT:

{filing_text}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=OLLAMA_TIMEOUT
    )

    response.raise_for_status()

    answer = (
        response.json()
        .get("message", {})
        .get("content", "")
    )

    result = json.loads(answer)

    products = result.get("products", [])

    if not isinstance(products, list):
        return []

    cleaned_products = []

    for product in products:

        if not isinstance(product, dict):
            continue

        name = str(
            product.get("new_product", "")
        ).strip()

        description = str(
            product.get("product_description", "")
        ).strip()

        if not name or not description:
            continue

        if name.lower() in [
            "none",
            "n/a",
            "not applicable",
            "no new product"
        ]:
            continue

        cleaned_products.append({
            "new_product": name[:200],
            "product_description": description[:179]
        })

    return cleaned_products


# ============================================================
# 8. MAIN WORKFLOW
# ============================================================

def main():

    # Validate email.
    if (
        not SEC_EMAIL
        or "@" not in SEC_EMAIL
        or not SEC_EMAIL.lower().endswith("@ucf.edu")
    ):
        raise ValueError(
            "Enter a valid UCF email in SEC_EMAIL."
        )

    print("=" * 60)
    print("SEC FORM 8-K NEW PRODUCT EXTRACTION")
    print("=" * 60)

    # Reset error log.
    error_path = BASE_DIR / ERROR_LOG

    if error_path.exists():
        error_path.unlink()

    session = get_session()

    # Step 1: Load S&P 500 companies.
    companies = get_sp500_companies()

    # Step 2: Check Ollama.
    test_ollama_connection()

    # Step 3: Retrieve SEC filings.
    print(
        "\nStep 3/5: Retrieving recent 8-K filings...",
        flush=True
    )

    all_filings = []

    for index, company in companies.iterrows():

        if len(all_filings) >= MAX_FILINGS_TO_ANALYZE:
            break

        remaining = (
            MAX_FILINGS_TO_ANALYZE - len(all_filings)
        )

        try:
            company_filings = get_recent_8k_filings(
                session=session,
                cik=company["cik"],
                ticker=company["ticker"],
                company_name=company["company_name"],
                remaining_filings=remaining
            )

            all_filings.extend(company_filings)

        except Exception as error:

            message = (
                f"Filing lookup failed for "
                f"{company['ticker']}: {error}"
            )

            print(message)
            log_error(message)

        time.sleep(SEC_PAUSE)

        if (index + 1) % 25 == 0:
            print(
                f"Companies checked: {index + 1}/"
                f"{len(companies)} | "
                f"8-K filings found: {len(all_filings)}",
                flush=True
            )

    all_filings = all_filings[
        :MAX_FILINGS_TO_ANALYZE
    ]

    print(
        f"\n8-K filings selected: {len(all_filings)}",
        flush=True
    )

    # Create filing review records.
    review_rows = []

    for filing in all_filings:
        review_rows.append({
            "company_name": filing["company_name"],
            "stock_name": filing["stock_name"],
            "filing_time": filing["filing_time"],
            "accession_number": filing["accession_number"],
            "filing_url": filing["filing_url"],
            "status": "Pending",
            "products_found": 0
        })

    review_df = pd.DataFrame(
        review_rows,
        columns=REVIEW_COLUMNS
    )

    review_path = BASE_DIR / REVIEW_CSV

    review_df.to_csv(
        review_path,
        index=False
    )

    # Step 4: Analyze filings using Ollama.
    print(
        "\nStep 4/5: Analyzing filings with Ollama...",
        flush=True
    )

    product_rows = []

    for index, filing in enumerate(all_filings):

        print(
            f"\nFiling {index + 1}/{len(all_filings)}: "
            f"{filing['stock_name']} - "
            f"{filing['filing_time']}",
            flush=True
        )

        try:
            filing_text = download_filing_text(
                session,
                filing["filing_url"]
            )

            if len(filing_text) < 200:
                raise ValueError(
                    "Filing text was too short to analyze."
                )

            products = extract_products_with_llm(
                filing_text,
                filing["company_name"]
            )

            for product in products:

                product_rows.append({
                    "company_name": filing["company_name"],
                    "stock_name": filing["stock_name"],
                    "filing_time": filing["filing_time"],
                    "new_product": product["new_product"],
                    "product_description": (
                        product["product_description"][:179]
                    )
                })

            review_df.loc[index, "status"] = "Analyzed"

            review_df.loc[
                index,
                "products_found"
            ] = len(products)

            print(
                f"Products identified: {len(products)}",
                flush=True
            )

        except Exception as error:

            message = (
                f"Analysis failed for "
                f"{filing['stock_name']} "
                f"{filing['accession_number']}: {error}"
            )

            print(message)
            log_error(message)

            review_df.loc[index, "status"] = "Error"

        # Save progress after every filing.
        products_df = pd.DataFrame(
            product_rows,
            columns=PRODUCT_COLUMNS
        )

        products_df.to_csv(
            BASE_DIR / OUTPUT_CSV,
            index=False
        )

        review_df.to_csv(
            review_path,
            index=False
        )

        time.sleep(OLLAMA_PAUSE)

    # Step 5: Save final CSV files.
    print(
        "\nStep 5/5: Saving final results...",
        flush=True
    )

    final_df = pd.DataFrame(
        product_rows,
        columns=PRODUCT_COLUMNS
    )

    if not final_df.empty:
        final_df = final_df.drop_duplicates(
            subset=[
                "company_name",
                "stock_name",
                "filing_time",
                "new_product"
            ]
        )

    final_df.to_csv(
        BASE_DIR / OUTPUT_CSV,
        index=False
    )

    review_df.to_csv(
        review_path,
        index=False
    )

    companies_represented = (
        final_df["stock_name"].nunique()
        if not final_df.empty
        else 0
    )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    print(f"Filings analyzed: {len(all_filings)}")
    print(f"Product rows extracted: {len(final_df)}")
    print(f"Companies represented: {companies_represented}")

    print(
        f"\nProduct CSV: {BASE_DIR / OUTPUT_CSV}"
    )

    print(
        f"Filing review CSV: {review_path}"
    )

    print(
        "\nReview the extracted products against their "
        "original SEC filings before submitting."
    )

    print(
        "The script does not guarantee 100 product rows. "
        "The number depends on actual product announcements."
    )


# ============================================================
# 9. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
