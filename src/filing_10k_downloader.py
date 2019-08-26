import os
from concurrent.futures.thread import ThreadPoolExecutor

from SECEdgar.crawler import SecCrawler
from shutil import move, rmtree

from src.cik_fetcher import fetch_cik_from_edgar
from src.tickers import TICKERS


def download_files_from_edgar(ticker, filing_letter):
    print('Downloading: ' + ticker + '...')

    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    filings_dir = os.path.join(current_file_dir_path, '..', 'data', '10' + filing_letter)
    if not os.path.exists(filings_dir):
        os.makedirs(filings_dir)

    cik = fetch_cik_from_edgar(ticker)
    if cik is None:
        return

    crawler = SecCrawler(filings_dir)
    download_method = getattr(crawler, 'filing_10' + filing_letter)
    download_method(ticker, cik, '20020101', '1000')

    src_dir_path = os.path.join(filings_dir, ticker, cik, '10-' + filing_letter)
    for file_name in os.listdir(src_dir_path):
        move(os.path.join(src_dir_path, file_name),
             os.path.join(filings_dir, ticker))

    rmtree(os.path.join(filings_dir, ticker, cik), ignore_errors=True)

    print('Done.')


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=8) as executor:
        for ticker in TICKERS:
            future = executor.submit(download_files_from_edgar, ticker, 'K')
