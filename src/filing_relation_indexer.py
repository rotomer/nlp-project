import os

from src.filing_8k_item_indexer import get_items_of_file
from src.stanford_nlp_invoker import index_relations_for_file

INTERESTING_ITEMS = {'Item 5.02'}


def _file_has_interesting_items(input_file_path):
    file_items_set = get_items_of_file(input_file_path)
    for item in file_items_set:
        if item in INTERESTING_ITEMS:
            return True

    return False


def index_relations(indexed_8k_dir, relations_8k_dir):
    skip_folder = 'True'
    for folder_name in os.listdir(indexed_8k_dir):
        ticker_indexed_8k_dir = os.path.join(indexed_8k_dir, folder_name)

        if skip_folder == 'True' and folder_name == 'PWR':
            skip_folder = 'Mid'

        if skip_folder == 'True':
            print('Skipping: ' + skip_folder)
            continue

        output_dir_path = os.path.join(relations_8k_dir, folder_name)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        skip_file = True if skip_folder == 'Mid' else False

        for file_name in os.listdir(ticker_indexed_8k_dir):
            if skip_file and file_name == file_name == '2002-05-23.txt':
                skip_folder = False
                skip_file = False

            input_file_path = os.path.join(ticker_indexed_8k_dir, file_name)
            if skip_file \
                    or not _file_has_interesting_items(input_file_path) \
                    or os.stat(input_file_path).st_size > 150000:
                print('Skipping: ' + file_name)
                continue
            else:
                print('Indexing: ' + input_file_path + '...')
                index_relations_for_file(input_file_path, output_dir_path)
                print('Done.')


if __name__ == '__main__':
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_8k_dir = os.path.join(current_file_dir_path, '..', 'data', 'Indexed_8K')
    relations_8k_dir = os.path.join(current_file_dir_path, '..', 'data', 'Relations_8k')

    index_relations(indexed_8k_dir, relations_8k_dir)
