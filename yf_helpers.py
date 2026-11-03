import io
import os
import ssl
import zipfile
from datetime import UTC, datetime

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

# Some environments (e.g. missing CA bundles) fail cert verification for
# urllib-based fetches such as pandas.read_excel(url). Assign the factory
# function itself (NOT the result of calling it), since urllib CALLS this hook.
ssl._create_default_https_context = ssl._create_unverified_context  # ty:ignore[invalid-assignment]


def yf_filename(ticker, start, end):
    start = start.replace("-", "")
    end = end.replace("-", "")
    # check if file exists
    if isinstance(ticker, list):
        name = f"{ticker[0]}-{ticker[-1]}[{len(ticker)}]-{start}-{end}"
    else:
        name = f"{ticker}-{start}-{end}"
    return f"data/{name}.csv"


def yf_downloader(ticker, start="-10y", end="now"):
    date_format = "%Y-%m-%d"
    # get datetime strings
    end = datetime.now(tz=UTC).strftime(date_format) if end == "now" else end
    if start.startswith("-"):
        endd = end.split("-")
        if start.endswith("y"):
            start = f"{int(endd[0]) + int(start[:-1])}-{endd[1]}-{endd[2]}"
        elif start.endswith("m"):
            start = f"{endd[0]}-{int(endd[1]) + int(start[:-1])}-{endd[2]}"
        elif start.endswith("d"):
            start = f"{endd[0]}-{endd[1]}-{int(endd[2]) + int(start[:-1])}"
        else:
            raise ValueError("Invalid start date format")

    filename = yf_filename(ticker, start, end)
    if not os.path.exists(filename):
        print(f"Data not exists, downloading to {filename}")
        # get data
        fdata = yf.download(tickers=ticker, start=start, end=end, progress=False)
        if fdata.empty:
            print("Data not exists")
            return None
        fdata_df = (
            fdata.melt(ignore_index=False)
            .reset_index()
            .pivot_table(index=["Date", "Ticker"], columns="Price", values="value")
            .reset_index()
            .rename(str.lower, axis="columns")
            .rename(columns={"ticker": "symbol", "adj close": "adjusted"})
        )
        # remove index name
        fdata_df.columns.name = None
        # format date column
        fdata_df.date = pd.to_datetime(fdata_df.date).dt.strftime(date_format)
        # save to csv
        fdata_df.to_csv(filename, index=False)
    else:
        print("Data already exists")
    return filename


def yf_reader(filename, start, end):
    date_format = "%Y-%m-%d"
    # read file
    fdata_df = pd.read_csv(filename, parse_dates=["date"], index_col=False)

    startd = pd.to_datetime(start).strftime(date_format)
    endd = pd.to_datetime(end).strftime(date_format)
    queryf = f'date >= "{startd}" and date <= "{endd}"'
    return fdata_df.query(queryf)


def calc_all_returns(fdata_df):
    return (
        fdata_df.assign(
            returns=lambda x: x.groupby("symbol")["adjusted"].pct_change(
                fill_method=None
            )
        )
        .get(["symbol", "date", "returns"])
        .dropna(subset="returns")
    )


DOW_CONSTITUENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "dow_jones_constituents.csv"
)


def get_dow_jones_indeces(refresh=False):
    """
    Return the list of Dow Jones (DIA) constituents.

    By default it reads the *pinned* list committed in
    ``data/dow_jones_constituents.csv`` so builds are offline and
    reproducible (the real index composition changes over time). Pass
    ``refresh=True`` to scrape the live SSGA holdings and rewrite the
    pinned file.
    """
    if not refresh:
        return pd.read_csv(DOW_CONSTITUENTS_FILE, dtype=str)["symbol"].dropna().tolist()

    url = (
        "https://www.ssga.com/us/en/institutional/etfs/library-content/"
        "products/fund-data/etfs/us/holdings-daily-us-en-dia.xlsx"
    )
    tickers = pd.read_excel(url, skiprows=4, nrows=30).get("Ticker").dropna().tolist()
    # Refresh the committed cache so future builds are deterministic again.
    pd.DataFrame({"symbol": tickers}).to_csv(DOW_CONSTITUENTS_FILE, index=False)
    return tickers


def get_dax30_indeces():
    # get the list of the constituents of the dax 30 are obtained through web scraping
    url = "https://es.finance.yahoo.com/quote/%5EGDAXI/components"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    symbols = []
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 404:
        print("Data not taken")
    elif response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        div = soup.find("div", class_="Ovx(a) W(100%)")
        table = (
            div.find("table", class_="W(100%) M(0) BdB Bdc($seperatorColor)")
            if div is not None
            else None
        )
        if table:
            for row in table.find_all("tr"):
                cells = row.find_all("td", class_="Py(10px) Ta(start) Pend(10px)")
                if cells:
                    cell_text = [cell.get_text() for cell in cells]
                    symbols.append(cell_text)
            if symbols:
                symbols = [item[0] for item in symbols]

    return symbols


def get_ff_research_data():
    url_zip_file = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    response = requests.get(url_zip_file, timeout=10)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_name = z.namelist()[0]
            first_blank_line = False
            column_names = ""
            values_dict = {}
            with z.open(file_name) as f:
                for linea in f:
                    linea = linea.strip()
                    if not first_blank_line:
                        if not linea:
                            first_blank_line = True
                            continue
                    elif not linea and first_blank_line:
                        break
                    else:
                        values_list = linea.decode("utf-8").split(",")
                        if column_names == "":
                            column_names = values_list
                            column_names.pop(0)
                            continue
                        values_dict[pd.to_datetime(values_list[0], format="%Y%m")] = [
                            pd.to_numeric(value) for value in values_list[1:]
                        ]

            return values_dict, column_names
    return None
