import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# ---------- CONFIG ----------
START_YEAR = 2014
END_YEAR = 2025

FINANCIAL_METRICS = [
    "Sales", "Expenses", "Other Income", "Operating Profit", "OPM %",
    "Interest", "Depreciation", "Profit before tax", "Tax %",
    "Net Profit", "EPS in Rs", "Dividend Payout %"
]

SHAREHOLDING_METRICS = ["FIIs", "DIIs", "Public", "Others"]

# ---------- CLEANING HELPERS ----------
def clean_numeric(val):
    """Remove symbols and convert to float."""
    if val in (None, "", "-", "--"):
        return None
    val = re.sub(r"[,%₹Rs\s]", "", val)
    try:
        return float(val)
    except ValueError:
        return None

def clean_percentage(val):
    """Handle values with '%'."""
    if isinstance(val, str) and "%" in val:
        val = val.replace("%", "")
    try:
        return float(val)
    except:
        return None

# ---------- SCRAPING FUNCTIONS ----------
def fetch_screener_page(ticker):
    url = f"https://www.screener.in/company/{ticker.upper()}/consolidated/"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch page for {ticker}, status: {resp.status_code}")
    return BeautifulSoup(resp.text, "html.parser")

def parse_financials(soup):
    """Parse yearly P&L table into a dataframe."""
    pl_section = soup.find("section", id="profit-loss")
    if not pl_section:
        raise Exception("Profit & Loss section not found")

    table = pl_section.find("table")
    headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")][1:]
    rows = {}
    for tr in table.find("tbody").find_all("tr"):
        row_title = tr.find("td").get_text(strip=True)
        if any(metric.lower() in row_title.lower() for metric in FINANCIAL_METRICS):
            values = [clean_numeric(td.get_text(strip=True)) for td in tr.find_all("td")[1:]]
            rows[row_title] = values

    fin_df = pd.DataFrame(rows, index=headers).T
    fin_df = fin_df.loc[:, ~fin_df.columns.duplicated()]
    fin_df = fin_df.apply(pd.to_numeric, errors="coerce")
    fin_df = fin_df.loc[:, [col for col in fin_df.columns if re.match(r"Mar \d{4}", col)]]
    return fin_df

def parse_shareholding(soup):
    """Parse yearly shareholding table."""
    shp_section = soup.find("section", id="shareholding")
    if not shp_section:
        raise Exception("Shareholding section not found")

    yearly_table = shp_section.find("div", id="yearly-shp").find("table")
    headers = [th.get_text(strip=True) for th in yearly_table.find("thead").find_all("th")][1:]
    rows = {}
    for tr in yearly_table.find("tbody").find_all("tr"):
        row_title = tr.find("td").get_text(strip=True)
        if any(m.lower() in row_title.lower() for m in SHAREHOLDING_METRICS):
            values = [clean_percentage(td.get_text(strip=True)) for td in tr.find_all("td")[1:]]
            rows[row_title] = values

    shp_df = pd.DataFrame(rows, index=headers).T
    shp_df = shp_df.loc[:, [col for col in shp_df.columns if re.match(r"Mar \d{4}", col)]]
    shp_df = shp_df.apply(pd.to_numeric, errors="coerce")
    return shp_df

# ---------- IMPUTATION ----------
def fill_missing_years(df):
    """Ensure years Mar 2014–Mar 2025 exist and apply back/forward fill."""
    all_years = [f"Mar {y}" for y in range(START_YEAR, END_YEAR + 1)]
    df = df.reindex(columns=all_years)
    df = df.apply(lambda s: s.fillna(method="ffill").fillna(method="bfill"), axis=1)
    return df

# ---------- MAIN PIPELINE ----------
def build_financial_dataset(ticker):
    soup = fetch_screener_page(ticker)

    fin_df = parse_financials(soup)
    # shp_df = parse_shareholding(soup)

    # combined_df = pd.concat([fin_df, shp_df])
    combined_df = pd.concat([fin_df])
    combined_df = fill_missing_years(combined_df)

    out_file = f"{ticker.upper()}_ML_Dataset.csv"
    combined_df.to_csv(out_file, float_format="%.2f")
    print(f"✅ Saved dataset: {out_file}")
    print(f"Shape: {combined_df.shape}")
    return combined_df

# ---------- RUN ----------
def run_webscrape(ticker):
    return build_financial_dataset(ticker)

if __name__ == "__main__":
    run_webscrape("ITC")
