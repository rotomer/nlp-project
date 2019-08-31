import os

from src.stanford_nlp_invoker import index_relations_for_file

if __name__ == '__main__':
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_8k_dir = os.path.join(current_file_dir_path, '..', 'data', 'Indexed_8K')
    sentiment_8k_dir = os.path.join(current_file_dir_path, '..', 'data', 'Relations_8k')

    skip_folder = 'True'
    for folder_name in os.listdir(indexed_8k_dir):
        ticker_indexed_8k_dir = os.path.join(indexed_8k_dir, folder_name)

        if folder_name == 'A':
            skip_folder = 'Mid'

        if skip_folder == 'True':
            print('Skipping: ' + skip_folder)
            continue

        output_dir_path = os.path.join(sentiment_8k_dir, folder_name)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        skip_file = True if skip_folder == 'Mid' else False

        for file_name in os.listdir(ticker_indexed_8k_dir):
            if skip_file and file_name == file_name == '2004-11-19.txt':
                skip_file = False

            if skip_file:
                print('Skipping: ' + file_name)
                continue
            else:
                input_file_path = os.path.join(ticker_indexed_8k_dir, file_name)
                print('Indexing: ' + input_file_path + '...')
                index_relations_for_file(input_file_path, output_dir_path)
                print('Done.')
