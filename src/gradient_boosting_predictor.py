import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Union, List, Tuple

import matplotlib.pylab as plt
import pandas as pd
from joblib import load, dump
from numpy.core.multiarray import ndarray
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.encoder import Encoder
from src.simple_encoder import SimpleEncoder


@dataclass(frozen=True)
class Model:
    classifier: GradientBoostingClassifier
    encoders: List[Encoder]


class GradientBoostingPredictor(object):

    def __init__(self,
                 model: Model):
        self._classifier = model.classifier
        self._encoders = model.encoders

    def predict(self,
                df: pd.DataFrame,
                feature_column_names: Iterable[str],
                target_column_name: str) -> pd.DataFrame:

        self._encode_data_frame(df, self._encoders)

        X, y = self._create_X_y_data_frames(df, feature_column_names, target_column_name)
        #output = self._classifier.predict(X)

        output = self._validate(X, y, self._classifier)

        return self._create_output_data_frame(df, y, output)

    @staticmethod
    def create_predictor_from_training(training_df: pd.DataFrame,
                                       feature_column_names: List[str],
                                       target_column_name: str,
                                       encoded_column_names: List[str],
                                       should_optimize: bool = False,
                                       should_train_test_split: bool = False,
                                       split_save_dir: str = None):
        """
        :param training_df: input data frame to train upon. See sample input csv and query for more info.
        :param feature_column_names: column names that should be used for the prediction.
        :param target_column_name: the column we wish to predict.
        :param encoded_column_names: String columns which require encoding.
        :param should_optimize: true iff should perform randomized search cross validation to find best hyper
        parameters.
        :param should_train_test_split: true iff should train on only part of the input set and use the rest for
        validation.
        :param split_save_dir: not None iff should save the split data set to disk. Has no effect if
        should_train_test_split is False.
        """

        encoders = GradientBoostingPredictor._fit_encoders(training_df, encoded_column_names)
        training_df = GradientBoostingPredictor._encode_data_frame(training_df, encoders)

        X, y = GradientBoostingPredictor._create_X_y_data_frames(training_df, feature_column_names, target_column_name)

        if should_train_test_split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
            classifier = GradientBoostingPredictor._fit_classification_model(X_train, y_train, should_optimize)
            GradientBoostingPredictor._validate(X_test, y_test, classifier)
            if split_save_dir is not None:
                X_train[target_column_name] = y_train
                X_test[target_column_name] = y_test
                X_train.to_csv(os.path.join(split_save_dir, 'train.csv'))
                X_test.to_csv(os.path.join(split_save_dir, 'test.csv'))

        else:
            classifier = GradientBoostingPredictor._fit_classification_model(X, y, should_optimize)

        return GradientBoostingPredictor(Model(classifier=classifier,
                                               encoders=encoders))

    @staticmethod
    def load_predictor_from_file(predictor_file_path: str):
        model = load(predictor_file_path)

        return GradientBoostingPredictor(model)

    def persist_predictor_to_disk(self,
                                  predictor_file_path: str):
        model = Model(classifier=self._classifier,
                      encoders=self._encoders)

        filename = Path(predictor_file_path)
        filename.touch(exist_ok=True)
        dump(model, filename)

    def show_feature_importance_plot(self,
                                     df: pd.DataFrame,
                                     feature_column_names: Iterable[str]):
        feature_cols = [col for col in df.columns if col in feature_column_names]
        feat_imp = pd.Series(self._classifier.feature_importances_, feature_cols).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

        plt.show()

    @staticmethod
    def _fit_encoders(training_df: pd.DataFrame,
                      encoded_column_names) -> List[Encoder]:
        return [SimpleEncoder.create_by_fitting(training_df, encoded_column_name, 'Encoded' + encoded_column_name)
                for encoded_column_name in encoded_column_names]

    @staticmethod
    def _encode_data_frame(df: pd.DataFrame,
                           encoders: List[Encoder]) -> pd.DataFrame:
        for encoder in encoders:
            df = encoder.encode_data_frame(df)

        return df

    @staticmethod
    def _fit_classification_model(X_train: pd.DataFrame,
                                  y_train: ndarray,
                                  should_optimize: bool) -> GradientBoostingClassifier:
        start = datetime.now()
        print('started fitting the regression.')
        if should_optimize:
            base_model = GradientBoostingClassifier()
            param_dist = {"learning_rate": [0.005, 0.01, 0.05],
                          "max_depth": [3, 4, 5],
                          "min_samples_leaf": [3, 4],
                          'min_samples_split': [2, 3],
                          'n_estimators': [500, 1000, 2000]}

            classifier = RandomizedSearchCV(base_model, param_dist, cv=10, n_iter=10, random_state=5, verbose=2,
                                            scoring='neg_mean_squared_error', n_jobs=4)
            classifier.fit(X_train, y_train)

            print(classifier.best_score_)
            print(classifier.best_params_)

            classifier = classifier.best_estimator_

        else:
            params = {'learning_rate': 0.005,
                      'max_depth': 5,
                      'min_samples_leaf': 3,
                      'min_samples_split': 2,
                      'n_estimators': 1000}
            classifier = GradientBoostingClassifier(**params)
            classifier.fit(X_train, y_train)

            print('finished fitting the regression. minutes elapsed: %.2f' %
                         ((datetime.now() - start).total_seconds() / 60))

        return classifier

    @staticmethod
    def _create_X_y_data_frames(df: pd.DataFrame,
                                feature_column_names: Iterable[str],
                                target_column_name: str) -> Tuple[pd.DataFrame, ndarray]:
        feature_cols = [col for col in df.columns if col in feature_column_names]
        target_cols = [col for col in df.columns if col in [target_column_name]]

        X = df[feature_cols]
        y = df[target_cols].values.ravel()

        return X, y

    @staticmethod
    def _validate(X_test: Union[pd.DataFrame, None],
                  y_test: Union[ndarray, None],
                  classifier: GradientBoostingClassifier):
        output = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, output)
        print('Accuracy: %.4f' % accuracy)

        precision = precision_score(y_test, output)
        print('Precision: %.4f' % precision)

        recall = recall_score(y_test, output)
        print('Recall: %.4f' % recall)

        f1 = f1_score(y_test, output)
        print('F1: %.4f' % f1)

        return output

    @staticmethod
    def _create_output_data_frame(df: pd.DataFrame,
                                  actual: ndarray,
                                  prediction_output: ndarray) -> pd.DataFrame:
        output_df = df.assign(Actual=pd.Series(actual).values,
                              Predicted=pd.Series(prediction_output).values)

        output_df['Predicted'] = output_df.apply(lambda row: round(row['Predicted']), axis=1)

        return output_df
