from abc import ABC, abstractmethod

import pandas as pd


class Encoder(ABC):
    @abstractmethod
    def encode_data_frame(self,
                          df: pd.DataFrame):
        pass

    @abstractmethod
    def column_to_encode(self):
        pass

    @abstractmethod
    def encoded_column(self):
        pass
