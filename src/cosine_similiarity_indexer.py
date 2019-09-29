import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_lemmatizer import lemmatize_text


def get_cosine_sim(*strs):
    if any(element is None or element.strip() == '' for element in strs):
        return [[0, 0],
                [0, 0]]

    vectors = [t for t in _get_vectors(*strs)]
    return cosine_similarity(vectors)


def _get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def index_cosine_for_ticker(cosine_index_file, filing_ticker_folder_path):
    """
    define the first file index retroactively as the cosine index of the first 2 file.
    """

    path_head_tail = os.path.split(filing_ticker_folder_path)
    ticker = path_head_tail[1]
    file_number = 0
    prev_file_content = None
    first_file_name = None
    for file_name in os.listdir(filing_ticker_folder_path):
        file_number += 1

        filing_file_path = os.path.join(filing_ticker_folder_path, file_name)
        with open(filing_file_path, 'r') as filing_file:
            content = lemmatize_text(filing_file.read())

        if file_number == 1:
            first_file_name = file_name
        else:
            cosine_matrix = get_cosine_sim(prev_file_content, content)
            # we're using invert cosine index here in order for empty files show up as similar to the previous files.
            invert_cosine_index = 0 if cosine_matrix[0][1] == 0 else 1-cosine_matrix[0][1]

            if file_number == 2:
                cosine_index_file.write(','.join([ticker, first_file_name, str(invert_cosine_index)]) + '\n')

            cosine_index_file.write(','.join([ticker, file_name, str(invert_cosine_index)]) + '\n')

        prev_file_content = content


def index_cosine_similarities():
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    filing_dir = os.path.join(current_file_dir_path, '..', 'input_data', 'Indexed_10K')
    cosine_dir = os.path.join(current_file_dir_path, '..', 'data', 'Cosine_10K')

    if not os.path.exists(cosine_dir):
        os.makedirs(cosine_dir)

    cosine_file_path = os.path.join(cosine_dir, 'cosine_10k.csv')

    with open(cosine_file_path, 'a') as cosine_file:
        cosine_file.write(','.join(['Ticker', 'FileName', 'Cosine_Index']) + '\n')

    for folder_name in os.listdir(filing_dir):
        with open(cosine_file_path, 'a') as cosine_file:
            folder_path = os.path.join(filing_dir, folder_name)
            print('Analyzing cosine for: ' + folder_path + '...')
            index_cosine_for_ticker(cosine_file, folder_path)
            print('Done.')


if __name__ == '__main__':
    index_cosine_similarities()
