from typing import Dict

import pandas as pd

from src.encoder import Encoder

catch_all = 'catchAll'


class SimpleEncoder(Encoder):
    def __init__(self,
                 encoding_dict: Dict[str, int],
                 column_to_encode: str,
                 encoded_column: str):
        self._encoding_dict = encoding_dict
        self._column_to_encode = column_to_encode
        self._encoded_column = encoded_column

    def encode_data_frame(self,
                          df: pd.DataFrame):
        df[self.encoded_column()] = df.apply(lambda row: self._get_encoded_value(row[self.column_to_encode()]), axis=1)

        return df

    def column_to_encode(self):
        return self._column_to_encode

    def encoded_column(self):
        return self._encoded_column

    @staticmethod
    def create_by_fitting(training_df: pd.DataFrame,
                          column_to_encode: str,
                          encoded_column: str):
        builder = SimpleEncoder.EncodingDictBuilder()
        for index, row in training_df.iterrows():
            builder.add_string(row[column_to_encode])

        return SimpleEncoder(encoding_dict=builder.build(),
                             column_to_encode=column_to_encode,
                             encoded_column=encoded_column)

    def _get_encoded_value(self,
                           string: str) -> int:
        if string not in self._encoding_dict:
            return self._encoding_dict[catch_all]

        return self._encoding_dict[string]

    class EncodingDictBuilder(object):
        def __init__(self):
            self._encoding_dict = {catch_all: 1}
            self._counter = 2

        def add_string(self,
                       string: str):
            if string not in self._encoding_dict:
                self._encoding_dict[string] = self._counter
                self._counter = self._counter + 1

        def build(self) -> Dict[str, int]:
            return self._encoding_dict
