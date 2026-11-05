"""
SEC 8-K Exhibit Collector
Downloads press-release exhibits (especially EX-99.1 / EX-99.2)
for filings already collected in sec_8k_filing_batches.

No Ollama or local LLM is used.
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

SEC_EMAIL = "la146090@ucf.edu"
MAX_EXHIBITS_PER_FILING = 2
MAX_EXHIBIT_CHARS = 14000
REQUEST_TIMEOUT = 40
PAUSE = 0.15

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "sec_8k_filing_batches"
OUTPUT_DIR = BASE_DIR / "sec_8k_exhibit_batches"

HEADERS = {
    "User-Agent": f"UCF SEC 8-K Research Project {SEC_EMAIL}",
    "Accept-Encoding": "gzip, deflate",
}

def clean_html(data):
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "ix:header", "ix:hidden"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)[:MAX_EXHIBIT_CHARS]

def get_accession_directory(filing_url):
    # The filing URL is .../edgar/data/CIK/ACCESSION/primary-document.htm
    return filing_url.rsplit("/", 1)[0] + "/"

def select_exhibits(items):
    candidates = []
    for item in items:
        name = item.get("name", "")
        typ = item.get("type", "").upper()
        desc = item.get("description", "")
        lower_name = name.lower()

        if not name or lower_name.endswith((".xsd", ".xml", ".jpg", ".png")):
            continue
        if typ.startswith("EX-99") or re.search(r"ex99[-_.]?[12]", lower_name):
            # Prefer 99.1, then 99.2; avoid huge non-text files.
            priority = 0 if ("99.1" in typ or re.search(r"ex99[-_.]?1", lower_name)) else 1
            candidates.append((priority, name, typ, desc))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[:MAX_EXHIBITS_PER_FILING]

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Could not find {INPUT_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    for old in OUTPUT_DIR.glob("exhibits_*.jsonl"):
        old.unlink()

    session = requests.Session()
    session.headers.update(HEADERS)

    filings = []
    for batch_file in sorted(INPUT_DIR.glob("batch_*.jsonl")):
        with open(batch_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["_source_batch"] = batch_file.name
                    filings.append(row)

    print(f"Loaded {len(filings)} collected filings.", flush=True)

    results = []
    errors = []
    for i, filing in enumerate(filings, start=1):
        ticker = filing.get("stock_name", "")
        accession_dir = get_accession_directory(filing["filing_url"])
        index_url = accession_dir + "index.json"

        print(f"{i}/{len(filings)} Checking exhibits: {ticker}", flush=True)
        output = {
            "company_name": filing.get("company_name", ""),
            "stock_name": ticker,
            "filing_time": filing.get("filing_time", ""),
            "accession_number": filing.get("accession_number", ""),
            "filing_url": filing.get("filing_url", ""),
            "exhibits": [],
        }

        try:
            response = session.get(index_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            items = response.json().get("directory", {}).get("item", [])
            selected = select_exhibits(items)

            for priority, filename, typ, desc in selected:
                exhibit_url = accession_dir + filename
                try:
                    exhibit_response = session.get(
                        exhibit_url, timeout=REQUEST_TIMEOUT
                    )
                    exhibit_response.raise_for_status()
                    text = clean_html(exhibit_response.content)
                    if len(text) > 150:
                        output["exhibits"].append({
                            "filename": filename,
                            "exhibit_type": typ,
                            "description": desc,
                            "url": exhibit_url,
                            "text": text,
                        })
                except Exception as exc:
                    errors.append({
                        "ticker": ticker,
                        "url": exhibit_url,
                        "error": str(exc),
                    })
                time.sleep(PAUSE)

        except Exception as exc:
            errors.append({
                "ticker": ticker,
                "url": index_url,
                "error": str(exc),
            })

        results.append(output)

        # Save every 5 filings, so completed work is preserved.
        if i % 5 == 0 or i == len(filings):
            batch_no = ((i - 1) // 5) + 1
            out_path = OUTPUT_DIR / f"exhibits_{batch_no:03d}.jsonl"
            with open(out_path, "w", encoding="utf-8") as out:
                for item in results[-min(5, len(results)):]:
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved {out_path.name}", flush=True)

        time.sleep(PAUSE)

    with open(OUTPUT_DIR / "exhibit_download_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Filings checked: {len(filings)}\n")
        f.write(f"Filings with at least one exhibit: {sum(bool(x['exhibits']) for x in results)}\n")
        f.write(f"Total exhibits downloaded: {sum(len(x['exhibits']) for x in results)}\n")
        f.write(f"Errors: {len(errors)}\n")

    if errors:
        with open(OUTPUT_DIR / "exhibit_download_errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)

    print("\nEXHIBIT COLLECTION COMPLETE", flush=True)
    print(f"Filings checked: {len(filings)}", flush=True)
    print(
        "Filings with exhibits: "
        f"{sum(bool(x['exhibits']) for x in results)}",
        flush=True,
    )
    print(
        "Exhibits downloaded: "
        f"{sum(len(x['exhibits']) for x in results)}",
        flush=True,
    )
    print(f"Output folder: {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
