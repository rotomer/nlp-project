import os
import re

from src.cik_fetcher import fetch_cik_from_edgar


def download_list_creator():
    with open(r'C:\code\nlp-project\data\temp\downloadlist.txt', 'w') as download_list_file:
        filing_10k_path = r'C:\code\nlp-project\data\10K'
        for ticker in os.listdir(filing_10k_path):
            cik = fetch_cik_from_edgar(ticker)
            cik_no_zeros = re.sub('\.[0]*', '.', cik)

            ticker_folder_path = os.path.join(filing_10k_path, ticker)
            for file_name in os.listdir(ticker_folder_path):
                base = str(file_name.split('.')[0])
                download_list_file.write(
                    ticker + '_' + base + ',' + 'edgar/data/' + cik_no_zeros + '/' + file_name + '\n')


if __name__ == '__main__':
    download_list_creator()
