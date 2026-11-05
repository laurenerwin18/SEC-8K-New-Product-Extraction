# SEC 8-K New Product Extraction

## Project Overview
This project analyzes SEC Form 8-K filings from S&P 500 companies to identify newly announced products and services using Python and a locally hosted Large Language Model (LLM).

## Objective
The goal is to extract information about newly announced products and services from company filings and organize the results into a structured CSV dataset.

## Tools and Technologies
- Python 3
- Pandas
- Requests
- BeautifulSoup
- SEC EDGAR database
- Ollama
- Llama 3.2 (3B)
- Visual Studio Code

## Methodology
1. Retrieved the list of S&P 500 companies.
2. Matched companies with SEC EDGAR company records.
3. Retrieved Form 8-K filings.
4. Downloaded and extracted filing text.
5. Used the Llama 3.2 model through Ollama to identify newly announced products and services.
6. Exported extracted information into CSV format.

## Output Files
- `sec_assignment_sp500_ucf.py` – Python extraction script.
- `sec_8k_new_products.csv` – Extracted product and service announcements.
- `sec_8k_filings_review.csv` – Filing review and processing results.
- `requirements.txt` – Python package dependencies.

## Results
Final results, including the number of filings analyzed and product announcements extracted, will be added after processing is complete.