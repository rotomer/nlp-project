import datetime
import os
from traceback import print_exc

import pandas as pd

from src.gradient_boosting_predictor import GradientBoostingPredictor


class EpsBasedPipeline(object):
    def __init__(self):
        pass

    def train(self, prediction_horizon_in_days):
        tickers = self._get_list_of_tickers_from_indexed_EPS()
        self._create_prediction_data_file(tickers, prediction_horizon_in_days)  # todo: go over multiple horizons

        # training_df = pd.read_csv(training_file_path, header=0)
        #
        # predictor = GradientBoostingPredictor.create_predictor_from_training(
        #     subset_training_df,
        #     feature_column_names,
        #     self._target_column_name,
        #     should_train_test_split=should_train_test_split)

    def predict(self):
        pass

    def _get_list_of_tickers_from_indexed_EPS(self):
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
        indexed_eps_dir = os.path.join(current_file_dir_path, '..', 'data', 'Indexed_EPS')
        file_names = os.listdir(indexed_eps_dir)

        return map(lambda x: os.path.splitext(x)[0], file_names)

    def _create_prediction_data_file(self, tickers, prediction_horizon_in_days):

        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
        prediction_data_dir = os.path.join(current_file_dir_path, '..', 'data', 'EPS_Based_Prediction_Data')

        if not os.path.exists(prediction_data_dir):
            os.makedirs(prediction_data_dir)

        prediction_data_file_path = os.path.join(prediction_data_dir,
                                                 'eps_based_' + str(prediction_horizon_in_days) + '.csv')
        with open(prediction_data_file_path, 'w') as prediction_data_file:
            prediction_data_file.write('Ticker,Date,EPS_Surprise_Percentage,Trend\n')

            valid_tickers_list = []
            for ticker in tickers:
                try:
                    print('Creating eps based prediction data for:' + ticker + '...')
                    self._append_rows_to_csv(prediction_data_file, ticker, prediction_horizon_in_days)
                    valid_tickers_list.append(ticker)
                    print('Done.')
                except FileNotFoundError:
                    pass
                    # print_exc()

            print(str(valid_tickers_list))


    def _append_rows_to_csv(self, prediction_data_file, ticker, prediction_horizon_in_days):
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))

        indexed_eps_dir = os.path.join(current_file_dir_path, '..', 'data', 'Indexed_Eps')
        indexed_eps_file_path = os.path.join(indexed_eps_dir, ticker + '.txt')

        price_history_dir = os.path.join(current_file_dir_path, '..', 'data', 'price_history')
        price_history_file_path = os.path.join(price_history_dir, ticker + '.csv')

        open_prices_dict, close_prices_dict = self._get_prices_for_ticker(price_history_file_path)

        with open(indexed_eps_file_path, 'r') as indexed_eps_file:
            for line in indexed_eps_file:
                splits = line.split(',')

                eps_date_str = splits[0]
                eps_date = datetime.datetime.strptime(eps_date_str, "%Y%m%d").date()
                try:
                    eps_surprise_percentage = float(splits[1])
                except ValueError:
                    continue

                base_price = self._find_price_before(close_prices_dict, eps_date)
                price_at_horizon = self._find_price_at_horizon(close_prices_dict, eps_date, prediction_horizon_in_days)
                if base_price is None or price_at_horizon is None or base_price == 0.0 or price_at_horizon == 0.0:
                    continue

                price_trend_percentage = price_at_horizon / base_price
                price_trend = 1 if price_trend_percentage >= 1 else 0

                prediction_data_file.write(ticker + ',' + str(eps_date) + ',' + str(eps_surprise_percentage) + ',' +
                                           str(price_trend) + '\n')

    def _get_prices_for_ticker(self, price_history_file_path):
        open_prices_dict = {}
        close_prices_dict = {}

        with open(price_history_file_path, 'r') as price_history_file:
            line_number = 0
            for line in price_history_file:
                line_number = line_number + 1
                if line_number == 1:
                    continue

                splits = line.split(',')

                price_date_str = splits[0]
                price_date = datetime.datetime.strptime(price_date_str, "%Y-%m-%d").date()

                open_price = float(splits[1])
                close_price = float(splits[4])

                open_prices_dict[price_date] = open_price
                close_prices_dict[price_date] = close_price

        return open_prices_dict, close_prices_dict

    def _find_price_before(self, prices_dict, eps_date):
        base_date = eps_date - datetime.timedelta(days=1)
        days_back = 1
        while prices_dict.get(base_date) is None:
            base_date = base_date - datetime.timedelta(days=1)

            days_back = days_back + 1
            if days_back == 30:
                break

        return prices_dict.get(base_date)

    def _find_price_at_horizon(self, prices_dict, eps_date, prediction_horizon_in_days):
        base_date = eps_date + datetime.timedelta(days=prediction_horizon_in_days)
        days_in_future = 5
        while prices_dict.get(base_date) is None:
            base_date = base_date + datetime.timedelta(days=1)

            days_in_future = days_in_future + 1
            if days_in_future == prediction_horizon_in_days + 30:
                break

        return prices_dict.get(base_date)


if __name__ == '__main__':
    eps_based_pipeline = EpsBasedPipeline()
    eps_based_pipeline.train(5)

