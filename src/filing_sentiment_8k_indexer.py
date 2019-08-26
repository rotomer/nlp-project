import os

from textblob import TextBlob


def index_sentiment_data(ticker, input_dir, output_dir):
    indexed_sentiment_file_path = os.path.join(output_dir, ticker + '.csv')
    with open(indexed_sentiment_file_path, 'w') as indexed_sentiment_file:

        indexed_sentiment_file.write('Date,Polarity,Subjectivity\n')

        for file_name in os.listdir(input_dir):

            input_file_path = os.path.join(input_dir, file_name)
            with open(input_file_path, 'r') as input_file:

                date_str = str(file_name.split('.')[0])

                text = TextBlob(input_file.read())
                polarity_str = '%.4f' % text.polarity
                subjectivity_str = '%.4f' % text.subjectivity
                indexed_sentiment_file.write(date_str + ',' + polarity_str + ',' + subjectivity_str + '\n')


if __name__ == '__main__':
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_8k_dir = os.path.join(current_file_dir_path, '..', 'data', 'Indexed_8K')
    sentiment_8k_dir = os.path.join(current_file_dir_path, '..', 'data', 'Sentiment_8K')

    if not os.path.exists(sentiment_8k_dir):
        os.makedirs(sentiment_8k_dir)

    for folder_name in os.listdir(indexed_8k_dir):
        print('Indexing: ' + folder_name + '...')
        input_dir_path = os.path.join(indexed_8k_dir, folder_name)
        index_sentiment_data(folder_name, input_dir_path, sentiment_8k_dir)
        print('Done.')
