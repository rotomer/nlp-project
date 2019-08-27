import datetime

import os

from src.tickers import TICKERS


def index_eps_data(ticker, input_file_path, output_dir):
    with open(input_file_path, 'r') as input_file:
        date = None
        text = None

        for line in input_file:
            if line.startswith('<DOCUMENT>'):
                date = None
                text = ''

            elif line.startswith('</DOCUMENT>'):
                _output_text_to_file(output_dir, ticker, date, text)

            elif line.startswith('TIME:'):
                date_time_str = line.split(':')[1]
                date_str = date_time_str[:8]
                date = datetime.datetime.strptime(date_str, "%Y%m%d").date()

            else:
                text = text + line


def _output_text_to_file(output_dir, ticker, date, text):
    if ticker not in TICKERS:
        return

    if date is None or \
            (datetime.date(year=2002, month=1, day=1) > date) or \
            (date > datetime.date(year=2013, month=1, day=1)):
        return

    ticker_indexed_8k_dir = os.path.join(output_dir, ticker)
    if not os.path.exists(ticker_indexed_8k_dir):
        os.makedirs(ticker_indexed_8k_dir)

    indexed_8k_file_path = os.path.join(ticker_indexed_8k_dir, str(date) + '.txt')
    with open(indexed_8k_file_path, 'w') as indexed_8k_file:
        indexed_8k_file.write(text)


if __name__ == '__main__':
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_8k_dir = os.path.join(current_file_dir_path, '..', 'data', 'Indexed_8K')
    input_8k_dir = os.path.join(current_file_dir_path, '..', 'data', '8K', '')

    for file_name in os.listdir(input_8k_dir):
        #if file_name == 'AAPL':
        print('Indexing: ' + file_name + '...')
        input_file_path = os.path.join(input_8k_dir, file_name)
        index_eps_data(file_name, input_file_path, indexed_8k_dir)
        print('Done.')
