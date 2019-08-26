import os

import pandas as pd

from src.gradient_boosting_predictor import GradientBoostingPredictor

feature_column_names = [
    'EncodedTicker',
    'EPS_Surprise_Percentage',
    'Days_Since_EPS',
    'Polarity',
    'Subjectivity'
]

target_column_name = 'Trend'

encoded_column_names = ['Ticker']


class FilingSentiment8KPipeline(object):
    def __init__(self):
        pass

    def train(self, prediction_horizon_in_days):
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
        sentiment_8k_prediction_data_dir = os.path.join(current_file_dir_path, '..', 'data',
                                                        'Sentiment_8K_Prediction_Data')
        training_file_path = os.path.join(sentiment_8k_prediction_data_dir,
                                          'sentiment_8k_' + str(prediction_horizon_in_days) + '.csv')

        training_df = pd.read_csv(training_file_path)

        predictor = GradientBoostingPredictor.create_predictor_from_training(
            training_df,
            feature_column_names,
            target_column_name,
            encoded_column_names=encoded_column_names,
            should_train_test_split=True)
        predictor.show_feature_importance_plot(training_df, feature_column_names)

    def predict(self):
        pass


if __name__ == '__main__':
    pipeline = FilingSentiment8KPipeline()
    pipeline.train(5)
