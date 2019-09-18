import os
from itertools import islice


class MasterDictionary(object):
    def __init__(self, master_dict):
        self._master_dict = master_dict
        self._sentiment_field_names = ['Negative',
                                       'Positive',
                                       'Uncertainty',
                                       'Litigious',
                                       'Constraining',
                                       'Superfluous',
                                       'Interesting',
                                       'Modal']
        self._default = [(field_name, 0) for field_name in self._sentiment_field_names]

    def sentiment_for_word(self, word):
        sentiment = self._master_dict.get(word)
        if sentiment is not None:
            return sentiment
        else:
            return self._default

    @staticmethod
    def from_file(master_dictionary_file_path):
        master_dict = {}
        with open(master_dictionary_file_path, 'r') as master_dictionary_file:
            line_number = 0
            for line in master_dictionary_file:
                line_number = line_number + 1
                if line_number == 1:
                    continue

                splits = line.strip().split(',')
                sentiment = [('Negative', 1 if int(splits[7]) != 0 else 0),
                             ('Positive', 1 if int(splits[8]) != 0 else 0),
                             ('Uncertainty', 1 if int(splits[9]) != 0 else 0),
                             ('Litigious', 1 if int(splits[10]) != 0 else 0),
                             ('Constraining', 1 if int(splits[11]) != 0 else 0),
                             ('Superfluous', 1 if int(splits[12]) != 0 else 0),
                             ('Interesting', 1 if int(splits[13]) != 0 else 0),
                             ('Modal', 1 if int(splits[14]) != 0 else 0)]
                master_dict[splits[0]] = sentiment

        return MasterDictionary(master_dict)

    @property
    def sentiment_field_names(self):
        return self._sentiment_field_names

    @property
    def inner_dict(self):
        return self._master_dict


def _take(n, iterable):
    return list(islice(iterable, n))


if __name__ == '__main__':
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    master_dict_file_path = os.path.join(current_file_dir_path, '..', 'data', 'Sentiment_Dictionary',
                                         'LoughranMcDonald_MasterDictionary_2018.csv')

    master_dict = MasterDictionary.from_file(master_dict_file_path)

    for key, value in _take(100, master_dict.inner_dict.items()):
        print(key + ':' + str(value))
