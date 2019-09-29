import datetime

import os


def index_10k_data(input_file_path, output_dir):
    base = os.path.basename(input_file_path)
    ticker = base.split('_')[0]

    ticker_output_dir = os.path.join(output_dir, ticker)
    if not os.path.exists(ticker_output_dir):
        os.makedirs(ticker_output_dir)

    date = None
    content = ''
    state = 'LOOKING_FOR_DATE'

    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            if state == 'LOOKING_FOR_DATE' and line.startswith('FILE DATE:'):
                splits = line.split(':')
                date_str = splits[len(splits) - 1].strip()
                date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                state = 'LOOKING_FOR_CONTENT'

            elif state == 'LOOKING_FOR_CONTENT' and line.startswith('<SECTION>'):
                state = 'INSIDE_CONTENT'

            elif state == 'INSIDE_CONTENT' and line.startswith('</SECTION>'):
                state = 'LOOKING_FOR_CONTENT'
                content = content + ' \n'

            elif state == 'INSIDE_CONTENT':
                content = content + line

    if date is not None and \
            datetime.date(year=2002, month=1, day=1) <= date <= datetime.date(year=2013, month=1, day=1):
        output_file_path = os.path.join(ticker_output_dir, str(date) + '.txt')
        with open(output_file_path, 'w') as output_file:
            output_file.write(content)


if __name__ == '__main__':
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_10k_dir = os.path.join(current_file_dir_path, '..', '..', 'input_data', 'Indexed_10K')
    input_10k_dir = os.path.join(current_file_dir_path, '..', '..', '10K-MDA-Section', 'statements')

    for file_name in os.listdir(input_10k_dir):
        if file_name != 'temp.txt' and file_name != 'DOWNLOADLOG.txt' and file_name != 'downloadlist.txt':
            print('Indexing: ' + file_name + '...')
            input_10k_file_path = os.path.join(input_10k_dir, file_name)
            index_10k_data(input_10k_file_path, indexed_10k_dir)
            print('Done.')
