import datetime
import os
from time import strptime

import pandas as pd

from src.gradient_boosting_predictor import GradientBoostingPredictor

feature_column_names_8k = [
    'EncodedTicker',

    'EPS_Surprise_Percentage',
    'Days_Since_EPS',

    # 'Polarity',
    # 'Subjectivity',
    #
    # 'Negative',
    # 'Positive',
    # 'Uncertainty',
    # 'Litigious',
    # 'Constraining',
    # 'Superfluous',
    # 'Interesting',
    # 'Modal',
    # 'WordCount',
    #
    # 'CEO_Departure',
    # 'CFO_Departure'
    #
    # 'Avg_Polarity',
    # 'Avg_Subjectivity',
    # 'Avg_Negative',
    # 'Avg_Positive',
    # 'Avg_Litigious',
    # 'Avg_Constraining',
    # 'Avg_Superfluous',
    # 'Avg_Modal',
    # 'Avg_WordCount',
    # 'Avg_CEO_Departure',
    # 'Avg_CFO_Departure'
]

feature_column_names_10k = [
    'EncodedTicker',

    'EPS_Surprise_Percentage',
    'Days_Since_EPS',
    #
    # 'Polarity',
    # 'Subjectivity',
    #
    # 'Negative',
    # 'Positive',
    # 'Uncertainty',
    # 'Litigious',
    # 'Constraining',
    # 'Superfluous',
    # 'Interesting',
    # 'Modal',
    # 'WordCount',
    #
    # 'Cosine'
    #
    # 'Avg_Polarity',
    # 'Avg_Subjectivity',
    # 'Avg_Negative',
    # 'Avg_Positive',
    # 'Avg_Litigious',
    # 'Avg_Constraining',
    # 'Avg_Superfluous',
    # 'Avg_Modal',
    # 'Avg_WordCount',
    # 'Avg_Cosine'

]

target_column_name = 'Trend_720'

encoded_column_names = ['Ticker']


class FullPipeline(object):
    def __init__(self):
        pass

    def train(self):
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
        training_file_path = os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames',
                                          'relations_cosine_specialized_sentiment.csv')

        training_df = pd.read_csv(training_file_path)
        training_8k_df = training_df[training_df['Filing_Type'] == 1].copy()

        training_10k_df = training_df[training_df['Filing_Type'] == 0].copy()

        predictor = GradientBoostingPredictor.create_predictor_from_training(
            training_8k_df,
            feature_column_names_8k,
            target_column_name,
            encoded_column_names=encoded_column_names,
            should_train_test_split=True)

        predictor = GradientBoostingPredictor.create_predictor_from_training(
            training_10k_df,
            feature_column_names_10k,
            target_column_name,
            encoded_column_names=encoded_column_names,
            should_train_test_split=True)
        # predictor.show_feature_importance_plot(training_df, feature_column_names)

    def predict(self):
        pass


if __name__ == '__main__':
    pipeline = FullPipeline()
    pipeline.train()
