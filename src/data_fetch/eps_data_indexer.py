from bs4 import BeautifulSoup
import os


def index_eps_data(input_file_path, output_dir):
    base = os.path.basename(input_file_path)
    date = os.path.splitext(base)[0]

    with open(input_file_path) as fd:
        soup = BeautifulSoup(fd, 'html.parser')
        tables = soup.findAll("table")
        table = tables[len(tables) - 1]

        for tr in table.contents:

            if hasattr(tr, 'contents') and 'finance.yahoo.com' in str(tr.contents):
                tds = tr.contents
                ticker = tds[1].contents[0].string
                surprise = tds[2].contents[0].string

                if ticker != 'CON.F':
                    indexed_eps_file_path = os.path.join(output_dir, ticker + '.txt')
                    with open(indexed_eps_file_path, 'a') as ticker_indexed_eps_file:
                        ticker_indexed_eps_file.write(date + ',' + surprise + '\n')


if __name__ == '__main__':
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_eps_dir = os.path.join(current_file_dir_path, '..', '..', 'input_data', 'Indexed_EPS')
    input_eps_dir = os.path.join(current_file_dir_path, '..', '..', 'data', 'EPS')

    if not os.path.exists(indexed_eps_dir):
        os.makedirs(indexed_eps_dir)

    for file_name in os.listdir(input_eps_dir):
        if file_name != 'eps.ser':
            print('Indexing: ' + file_name + '...')
            input_eps_file_path = os.path.join(input_eps_dir, file_name)
            index_eps_data(input_eps_file_path, indexed_eps_dir)
            print('Done.')
