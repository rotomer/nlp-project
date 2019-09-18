import os

from textblob import TextBlob


def _index_sentiment_data(ticker, input_dir, output_dir):
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


def index_sentiment(filing_number):
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_filing_dir = os.path.join(current_file_dir_path, '..', 'data', 'Indexed_' + filing_number + 'K')
    sentiment_filing_dir = os.path.join(current_file_dir_path, '..', 'data', 'Sentiment_' + filing_number + 'K')

    if not os.path.exists(sentiment_filing_dir):
        os.makedirs(sentiment_filing_dir)

    for folder_name in os.listdir(indexed_filing_dir):
        print('Indexing: ' + folder_name + '...')
        input_dir_path = os.path.join(indexed_filing_dir, folder_name)
        _index_sentiment_data(folder_name, input_dir_path, sentiment_filing_dir)
        print('Done.')


if __name__ == '__main__':
    index_sentiment('10')
