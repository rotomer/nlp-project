from datetime import datetime
import os

from src.filing_8k_items import ITEMS_DICT, ITEMS_LIST


def _get_item_in_line(line):
    for item_str in ITEMS_DICT.keys():
        if item_str.lower() in line.lower():
            return ITEMS_DICT[item_str]

    return None


def get_items_of_file(input_file_path):
    file_items = set()

    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            item = _get_item_in_line(line)

            if item is not None:
                file_items.add(item)

    return file_items


def index_items_of_all_8ks(filing_8k_folder_path, items_index_folder_path):
    if not os.path.exists(items_index_folder_path):
        os.makedirs(items_index_folder_path)

    items_index_file_path = os.path.join(items_index_folder_path, '8k_items.csv')
    with open(items_index_file_path, 'w') as items_index_file:
        items_index_file.write('key,' + ','.join(ITEMS_LIST) + '\n')

        for ticker_folder_name in os.listdir(filing_8k_folder_path):
            print('Indexing items for: ' + ticker_folder_name + '...')

            # if ticker_folder_name != 'AAPL':
            #     continue

            ticker_folder_path = os.path.join(filing_8k_folder_path, ticker_folder_name)
            for filing_8_file_name in os.listdir(ticker_folder_path):
                filing_8_file_path = os.path.join(ticker_folder_path, filing_8_file_name)

                file_items = get_items_of_file(filing_8_file_path)
                items_bit_vector = ['1' if item in file_items else '0' for item in ITEMS_LIST]
                file_key = ticker_folder_name + '_' + filing_8_file_name
                items_index_file.write(file_key + ',' + ','.join(items_bit_vector) + '\n')

            print('Done.')


def load_items_from_index(items_index_file_path):
    items_dict = {} # file_key -> set of items
    with open(items_index_file_path, 'r') as items_index_file:
        line_number = 0
        for line in items_index_file:
            line_number = line_number + 1
            if line_number == 1:
                continue

            splits = line.strip().split(',')
            file_key = splits[0]

            items_bit_vector = splits[1:]
            inner_set = set()
            for i in range(len(ITEMS_LIST)):
                if items_bit_vector[i] == '1':
                    inner_set.add(ITEMS_LIST[i])

            items_dict[file_key] = inner_set

    return items_dict


if __name__ == '__main__':
    #print(get_items_of_file(r'C:\code\nlp-project\data\Indexed_8K\LXK\2009-02-26.txt'))

    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_8k_dir = os.path.join(current_file_dir_path, '..', 'data', 'Indexed_8K')
    items_8k_dir = os.path.join(current_file_dir_path, '..', 'data', 'Items_8K')

    #index_items_of_all_8ks(indexed_8k_dir, items_8k_dir)

    start = datetime.now()
    items_dict = load_items_from_index(os.path.join(items_8k_dir, '8k_items.csv'))
    print(str((datetime.now() - start).seconds))

