import os

import nltk

from src.master_dictionary import MasterDictionary


def _columns(master_dict):
    return ['Ticker'] + \
           ['FileName'] + \
           master_dict.sentiment_field_names + \
           ['Relative' + sentiment_field for sentiment_field in master_dict.sentiment_field_names] + \
           ['WordCount']


def _specialized_sentiment_analysis_for_file(input_filing_file_path, output_sentiment_file, master_dict):
    counters = {sentiment_field: 0 for sentiment_field in master_dict.sentiment_field_names}
    num_words = 0
    with open(input_filing_file_path, 'r') as input_filing_file:
        for line in input_filing_file:
            tokens = nltk.word_tokenize(line)
            num_words = num_words + len(tokens)
            for token in tokens:
                sentiment_record = master_dict.sentiment_for_word(token)
                for sentiment_field, sentiment_bit in sentiment_record:
                    counters[sentiment_field] = counters[sentiment_field] + sentiment_bit

    relative_counters = {'Relative' + sentiment_field: counters[sentiment_field] / (num_words if num_words != 0 else 1)
                         for sentiment_field in master_dict.sentiment_field_names}
    counters.update(relative_counters)
    counters['WordCount'] = num_words
    path_head_tail = os.path.split(input_filing_file_path)
    counters['FileName'] = path_head_tail[1]
    counters['Ticker'] = os.path.split(path_head_tail[0])[1]

    output_sentiment_file.write(','.join(str(counters[column]) for column in _columns(master_dict)) + '\n')


def index_specialized_sentiment_analysis(filing_number):
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    specialized_sentiment_dir = os.path.join(current_file_dir_path, '..', 'data', 'Specialized_Sentiment')
    filing_dir = os.path.join(current_file_dir_path, '..', 'input_data', 'Indexed_' + filing_number + 'K')

    master_dict_file_path = os.path.join(current_file_dir_path, '..', 'input_data', 'Sentiment_Dictionary',
                                         'LoughranMcDonald_MasterDictionary_2018.csv')
    master_dict = MasterDictionary.from_file(master_dict_file_path)

    if not os.path.exists(specialized_sentiment_dir):
        os.makedirs(specialized_sentiment_dir)

    sentiment_file_path = os.path.join(specialized_sentiment_dir, 'Specialized_Sentiment_' + filing_number + 'k.csv')
    with open(sentiment_file_path, 'w') as sentiment_file:
        sentiment_file.write(','.join(_columns(master_dict)) + '\n')

        for folder_name in os.listdir(filing_dir):
            folder_path = os.path.join(filing_dir, folder_name)
            for file_name in os.listdir(folder_path):
                filing_file_path = os.path.join(folder_path, file_name)
                print('Analyzing sentiment for: ' + filing_file_path + '...')
                _specialized_sentiment_analysis_for_file(filing_file_path, sentiment_file, master_dict)
                print('Done.')


if __name__ == '__main__':
    index_specialized_sentiment_analysis('10')
