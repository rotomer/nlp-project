import datetime
import os


def create_sentiment_8k_prediction_data(prediction_horizon_in_days):
    tickers = _get_list_of_tickers_from_sentiment_8k()
    _create_prediction_data_file(tickers, prediction_horizon_in_days)


def _get_list_of_tickers_from_sentiment_8k():
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    sentiment_8k_dir = os.path.join(current_file_dir_path, '..', 'data', 'Sentiment_8K')
    file_names = os.listdir(sentiment_8k_dir)

    return map(lambda x: os.path.splitext(x)[0], file_names)


def _create_prediction_data_file(tickers, prediction_horizon_in_days):
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    prediction_data_dir = os.path.join(current_file_dir_path, '..', 'data', 'Sentiment_8K_Prediction_Data')

    if not os.path.exists(prediction_data_dir):
        os.makedirs(prediction_data_dir)

    prediction_data_file_path = os.path.join(prediction_data_dir,
                                             'sentiment_8k_' + str(prediction_horizon_in_days) + '.csv')
    with open(prediction_data_file_path, 'w') as prediction_data_file:
        prediction_data_file.write('Ticker,Date,EPS_Surprise_Percentage,Days_Since_EPS,Polarity,Subjectivity,Trend\n')

        valid_tickers_list = []
        for ticker in tickers:
            try:
                print('Creating sentiment based 8k prediction data for:' + ticker + '...')
                _append_rows_to_csv(prediction_data_file, ticker, prediction_horizon_in_days)
                valid_tickers_list.append(ticker)
                print('Done.')
            except FileNotFoundError:
                pass
                # print_exc()

        print(str(valid_tickers_list))


def _append_rows_to_csv(prediction_data_file, ticker, prediction_horizon_in_days):
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))

    indexed_eps_dir = os.path.join(current_file_dir_path, '..', 'input_data', 'Indexed_Eps')
    indexed_eps_file_path = os.path.join(indexed_eps_dir, ticker + '.txt')

    price_history_dir = os.path.join(current_file_dir_path, '..', 'input_data', 'price_history')
    price_history_file_path = os.path.join(price_history_dir, ticker + '.csv')

    sentiment_8k_dir = os.path.join(current_file_dir_path, '..', 'data', 'Sentiment_8K')
    sentiment_8k_file_path = os.path.join(sentiment_8k_dir, ticker + '.csv')

    open_prices_dict, close_prices_dict = _get_prices_for_ticker(price_history_file_path)
    eps_list = _get_eps_list_from_file(indexed_eps_file_path)

    line_number = 0
    with open(sentiment_8k_file_path, 'r') as sentiment_8k_file:
        for line in sentiment_8k_file:
            line_number = line_number + 1
            if line_number == 1:
                continue

            splits = line.split(',')

            filing_date_str = splits[0]
            filing_date = datetime.datetime.strptime(filing_date_str, "%Y-%m-%d").date()
            try:
                polarity = float(splits[1])
                subjectivity = float(splits[2])
            except ValueError:
                continue

            try:
                eps_surprise_percentage, days_since_eps = _get_relevant_eps_data(filing_date, eps_list)
            except ValueError:
                continue

            base_price = _find_base_price(close_prices_dict, filing_date)
            price_at_horizon = _find_price_at_horizon(close_prices_dict, filing_date, prediction_horizon_in_days)
            if base_price is None or price_at_horizon is None or base_price == 0.0 or price_at_horizon == 0.0:
                continue

            price_trend_percentage = price_at_horizon / base_price
            price_trend = 1 if price_trend_percentage >= 1 else 0

            prediction_data_file.write(
                ticker + ',' + str(filing_date) + ',' + str(eps_surprise_percentage) + ',' + str(days_since_eps) + ','
                + str(polarity) + ',' + str(subjectivity) + ',' + str(price_trend) + '\n')


def _get_relevant_eps_data(filing_date, eps_list):
    for eps_date, eps_surprise_percentage in eps_list:
        if eps_date <= filing_date:
            return eps_surprise_percentage, (filing_date - eps_date).days

    raise ValueError


def _get_eps_list_from_file(indexed_eps_file_path):
    eps_list = []

    with open(indexed_eps_file_path, 'r') as indexed_eps_file:
        for line in indexed_eps_file:
            splits = line.split(',')

            eps_date_str = splits[0]
            eps_date = datetime.datetime.strptime(eps_date_str, "%Y%m%d").date()
            try:
                eps_surprise_percentage = float(splits[1])
            except ValueError:
                continue

            eps_list.append((eps_date, eps_surprise_percentage))

    return sorted(eps_list, key=lambda x: x[0], reverse=True)


def _get_prices_for_ticker(price_history_file_path):
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


def _find_base_price(prices_dict, eps_date):
    base_date = eps_date - datetime.timedelta(days=1)
    days_back = 1
    while prices_dict.get(base_date) is None:
        base_date = base_date - datetime.timedelta(days=1)

        days_back = days_back + 1
        if days_back == 30:
            break

    return prices_dict.get(base_date)


def _find_price_at_horizon(prices_dict, eps_date, prediction_horizon_in_days):
    base_date = eps_date + datetime.timedelta(days=prediction_horizon_in_days)
    days_in_future = 5
    while prices_dict.get(base_date) is None:
        base_date = base_date + datetime.timedelta(days=1)

        days_in_future = days_in_future + 1
        if days_in_future == prediction_horizon_in_days + 30:
            break

    return prices_dict.get(base_date)


if __name__ == '__main__':
    create_sentiment_8k_prediction_data(5)
