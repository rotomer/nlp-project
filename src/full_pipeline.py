import os

import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar

from src.gradient_boosting_predictor import GradientBoostingPredictor

eps_feature_column_names = [
    'EncodedTicker',

    'EPS_Surprise_Percentage',
    'Days_Since_EPS'
]

feature_column_names_8k = [
    'EncodedTicker',

    'EPS_Surprise_Percentage',
    'Days_Since_EPS',

    'Polarity',
    'Subjectivity',

    'Negative',
    'Positive',
    'Uncertainty',
    'Litigious',
    'Constraining',
    'Superfluous',
    'Interesting',
    'Modal',
    'WordCount',

    'CEO_Departure',
    'CFO_Departure'

    'Avg_Polarity',
    'Avg_Subjectivity',
    'Avg_Negative',
    'Avg_Positive',
    'Avg_Litigious',
    'Avg_Constraining',
    'Avg_Superfluous',
    'Avg_Modal',
    'Avg_WordCount',
    'Avg_CEO_Departure',
    'Avg_CFO_Departure'
]

feature_column_names_10k = [
    'EncodedTicker',

    'EPS_Surprise_Percentage',
    'Days_Since_EPS',

    'Polarity',
    'Subjectivity',

    'Negative',
    'Positive',
    'Uncertainty',
    'Litigious',
    'Constraining',
    'Superfluous',
    'Interesting',
    'Modal',
    'WordCount',

    'Cosine'

    'Avg_Polarity',
    'Avg_Subjectivity',
    'Avg_Negative',
    'Avg_Positive',
    'Avg_Litigious',
    'Avg_Constraining',
    'Avg_Superfluous',
    'Avg_Modal',
    'Avg_WordCount',
    'Avg_Cosine'

]

target_column_name = 'Trend_1080'

encoded_column_names = ['Ticker']


class FullPipeline(object):
    def __init__(self):
        pass

    def split(self, target_column):
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
        training_file_path = os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames',
                                          'relations_cosine_specialized_sentiment_1080.csv')

        df = pd.read_csv(training_file_path)
        y = df[target_column].values.ravel()

        train_df, test_df = train_test_split(df, test_size=0.2, stratified=y)

        train_df.to_csv(os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '10K',
                                     'train_7.csv'))

        test_df.to_csv(os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '10K',
                                    'test_7.csv'))

    @staticmethod
    def train_test_inner_8k(horizon):
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))

        train_file_path_8k = os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '8K',
                                          'train.csv')
        test_file_path_8k = os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '8K',
                                         'test.csv')
        train_8k_df = pd.read_csv(train_file_path_8k)
        test_8k_df = pd.read_csv(test_file_path_8k)

        target_col_name = 'Trend_' + str(horizon)
        predictor = GradientBoostingPredictor.create_predictor_from_training(
            train_8k_df,
            feature_column_names_8k,
            target_col_name,
            encoded_column_names=encoded_column_names)

        output_df = predictor.predict(test_8k_df, feature_column_names_8k, target_col_name)

        predictor = GradientBoostingPredictor.create_predictor_from_training(
            train_8k_df,
            eps_feature_column_names,
            target_col_name,
            encoded_column_names=encoded_column_names)

        output_df = predictor.predict(test_8k_df, eps_feature_column_names, target_col_name)

    @staticmethod
    def train_test_inner_10k(horizon):
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))

        train_file_path_10k = os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '10K',
                                           'train.csv')
        test_file_path_10k = os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '10K',
                                          'test.csv')
        train_10k_df = pd.read_csv(train_file_path_10k)
        test_10k_df = pd.read_csv(test_file_path_10k)

        target_col_name = 'Trend_' + str(horizon)
        predictor = GradientBoostingPredictor.create_predictor_from_training(
            train_10k_df,
            feature_column_names_10k,
            target_col_name,
            encoded_column_names=encoded_column_names)

        output_df = predictor.predict(test_10k_df, feature_column_names_10k, target_col_name)

        predictor = GradientBoostingPredictor.create_predictor_from_training(
            train_10k_df,
            eps_feature_column_names,
            target_col_name,
            encoded_column_names=encoded_column_names)

        output_df = predictor.predict(test_10k_df, eps_feature_column_names, target_col_name)

    def train_and_test_8k(self):
        # with ProcessPoolExecutor(max_workers=8) as executor:
        for horizon in [5, 30, 90, 180, 360, 720, 1080]:
            print("horizon:" + str(horizon))
            FullPipeline.train_test_inner_8k(horizon)
            print('----')

    def train_and_test_10k(self):
        # with ProcessPoolExecutor(max_workers=8) as executor:
        for horizon in [5, 30, 90, 180, 360, 720, 1080]:
            print("horizon:" + str(horizon))
            FullPipeline.train_test_inner_10k(horizon)
            print('----')

    @staticmethod
    def calc_mcnemar_8k(horizon):
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))

        train_file_path_8k = os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '8K',
                                          'train.csv')
        test_file_path_8k = os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '8K',
                                         'test.csv')

        test_8k_df = pd.read_csv(test_file_path_8k)

        target_col_name = 'Trend_' + str(horizon)

        predictor = GradientBoostingPredictor.load_predictor_from_file(
            os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '8K',
                         'model_ling' + str(horizon) + '.joblib'))

        ling_output_df = predictor.predict(test_8k_df, feature_column_names_8k, target_col_name)

        predictor = GradientBoostingPredictor.load_predictor_from_file(
            os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '8K',
                         'model_eps' + str(horizon) + '.joblib'))
        eps_output_df = predictor.predict(test_8k_df, eps_feature_column_names, target_col_name)

        contingency_table = FullPipeline.calc_contingency_table(ling_output_df, eps_output_df)
        print(str(contingency_table))

        result = mcnemar(contingency_table, exact=True)
        print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
        alpha = 0.05
        if result.pvalue > alpha:
            print('Same proportions of errors (fail to reject H0)')
        else:
            print('Different proportions of errors (reject H0)')

        print(str(contingency_table))

    @staticmethod
    def calc_mcnemar_10k(horizon):
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))

        train_file_path_10k = os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '10K',
                                          'train.csv')
        test_file_path_10k = os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '10K',
                                         'test.csv')
        train_10k_df = pd.read_csv(train_file_path_10k)
        test_10k_df = pd.read_csv(test_file_path_10k)

        target_col_name = 'Trend_' + str(horizon)

        predictor = GradientBoostingPredictor.load_predictor_from_file(
            os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '10K',
                         'model_ling' + str(horizon) + '.joblib'))

        ling_output_df = predictor.predict(test_10k_df, feature_column_names_10k, target_col_name)

        predictor = GradientBoostingPredictor.load_predictor_from_file(
            os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames', '10K',
                         'model_eps' + str(horizon) + '.joblib'))
        eps_output_df = predictor.predict(test_10k_df, eps_feature_column_names, target_col_name)

        contingency_table = FullPipeline.calc_contingency_table(ling_output_df, eps_output_df)
        print(str(contingency_table))

        # calculate mcnemar test
        result = mcnemar(contingency_table, exact=True)
        # summarize the finding
        print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
        # interpret the p-value
        alpha = 0.05
        if result.pvalue > alpha:
            print('Same proportions of errors (fail to reject H0)')
        else:
            print('Different proportions of errors (reject H0)')

        print(str(contingency_table))

    @staticmethod
    def calc_contingency_table(ling_output_df, eps_output_df):
        yes_ling_yes_eps = 0  # [0,0] (row, col)
        yes_ling_no_eps = 0  # [1,0] (row, col)
        no_ling_yes_eps = 0  # [0,1] (row, col)
        no_ling_no_eps = 0  # [1,1] (row, col)

        for index, _ in ling_output_df.iterrows():
            if ling_output_df.iloc[index, :].loc['Predicted'] == ling_output_df.iloc[index, :].loc['Actual'] and \
                    eps_output_df.iloc[index, :].loc['Predicted'] == eps_output_df.iloc[index, :].loc['Actual']:
                yes_ling_yes_eps += 1
            elif ling_output_df.iloc[index, :].loc['Predicted'] == ling_output_df.iloc[index, :].loc['Actual'] and \
                    eps_output_df.iloc[index, :].loc['Predicted'] != eps_output_df.iloc[index, :].loc['Actual']:
                yes_ling_no_eps += 1
            elif ling_output_df.iloc[index, :].loc['Predicted'] != ling_output_df.iloc[index, :].loc['Actual'] and \
                    eps_output_df.iloc[index, :].loc['Predicted'] == eps_output_df.iloc[index, :].loc['Actual']:
                no_ling_yes_eps += 1
            else:
                no_ling_no_eps += 1

        return [[yes_ling_yes_eps, no_ling_yes_eps],
                [yes_ling_no_eps, no_ling_no_eps]]


if __name__ == '__main__':
    pipeline = FullPipeline()
    # pipeline.train_and_test_8k()
    # pipeline.train_and_test_10k()
    # FullPipeline.train_test_inner_8k(720)
    # FullPipeline.train_test_inner_10k(1080)
    FullPipeline.calc_mcnemar_8k(1080)
    #pipeline.split(target_column_name)
