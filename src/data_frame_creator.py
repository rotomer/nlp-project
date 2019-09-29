import datetime
import os

from src import management_departure_indexer
from src.tickers import TICKERS

current_file_dir_path = os.path.dirname(os.path.realpath(__file__))


def _get_relevant_eps_data(filing_date, eps_list):
    for eps_date, eps_surprise_percentage in eps_list:
        if eps_date <= filing_date:
            return eps_surprise_percentage, (filing_date - eps_date).days

    raise ValueError


def _get_eps_list_from_file(ticker):
    indexed_eps_dir = os.path.join(current_file_dir_path, '..', 'input_data', 'Indexed_Eps')
    indexed_eps_file_path = os.path.join(indexed_eps_dir, ticker + '.txt')

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


horizons = [5, 30, 90, 180, 360, 720, 1080]

columns = ['Ticker',
           'Date',

           'Filing_Type',  # 0 for 10K, 1 for 8K

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

           'Cosine',

           'CEO_Departure',
           'CFO_Departure',

           'Avg_Polarity',
           'Avg_Subjectivity',
           'Avg_Negative',
           'Avg_Positive',
           'Avg_Litigious',
           'Avg_Constraining',
           'Avg_Superfluous',
           'Avg_Modal',
           'Avg_WordCount',
           'Avg_Cosine',
           'Avg_CEO_Departure',
           'Avg_CFO_Departure'

           ] + \
          ['Trend_' + str(horizon) for horizon in horizons]


def _get_prices_for_ticker(ticker):
    price_history_dir = os.path.join(current_file_dir_path, '..', 'input_data', 'price_history')
    price_history_file_path = os.path.join(price_history_dir, ticker + '.csv')

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


def _get_specialized_sentiment(filing_number):
    sentiment_file_path = os.path.join(current_file_dir_path, '..', 'data', 'Specialized_Sentiment',
                                       'Specialized_Sentiment_' + filing_number + 'k.csv')

    specialized_sentiments_per_ticker = {}
    line_number = 0
    last_ticker = ''
    ticker_sentiments_by_date = {}
    with open(sentiment_file_path, 'r') as sentiment_file:
        for line in sentiment_file:
            line_number = line_number + 1
            if line_number == 1:
                continue

            line = line.strip()
            if line == '':
                continue

            splits = line.split(',')

            ticker = splits[0]

            if last_ticker != ticker:
                if len(ticker_sentiments_by_date) > 0:
                    specialized_sentiments_per_ticker[last_ticker] = ticker_sentiments_by_date
                ticker_sentiments_by_date = {}
                last_ticker = ticker

            file_name = splits[1]
            filing_date_str = file_name.split('.')[0]
            filing_date = datetime.datetime.strptime(filing_date_str, "%Y-%m-%d").date()
            negative_count = int(splits[2])
            positive_count = int(splits[3])
            uncertainty_count = int(splits[4])
            litigious_count = int(splits[5])
            constraining_count = int(splits[6])
            superfluous_count = int(splits[7])
            interesting_count = int(splits[8])
            modal_count = int(splits[9])
            word_count = int(splits[18])
            record = (ticker, filing_date, negative_count, positive_count, uncertainty_count, litigious_count,
                      constraining_count, superfluous_count, interesting_count, modal_count, word_count)
            ticker_sentiments_by_date[filing_date] = record

    return specialized_sentiments_per_ticker


def _get_8k_specialized_sentiment():
    return _get_specialized_sentiment('8')


def _get_10k_specialized_sentiment():
    return _get_specialized_sentiment('10')


def _get_cosine_index_for_10k():
    cosine_file_path = os.path.join(current_file_dir_path, '..', 'data', 'Cosine_10K', 'cosine_10k.csv')

    cosine_indices_per_ticker = {}
    line_number = 0
    last_ticker = ''
    cosine_index_by_date = {}
    with open(cosine_file_path, 'r') as cosine_file:
        for line in cosine_file:
            line_number = line_number + 1
            if line_number == 1:
                continue

            line = line.strip()
            if line == '':
                continue

            splits = line.split(',')

            ticker = splits[0]

            if last_ticker != ticker:
                if len(cosine_index_by_date) > 0:
                    cosine_indices_per_ticker[last_ticker] = cosine_index_by_date
                cosine_index_by_date = {}
                last_ticker = ticker

            file_name = splits[1]
            filing_date_str = file_name.split('.')[0]
            filing_date = datetime.datetime.strptime(filing_date_str, "%Y-%m-%d").date()
            cosine_index = float(splits[2])
            cosine_index_by_date[filing_date] = cosine_index

    return cosine_indices_per_ticker


def _get_filing_sentiment_for_ticker(ticker, filing_number):
    sentiment_dir = os.path.join(current_file_dir_path, '..', 'data', 'Sentiment_' + filing_number + 'K')
    sentiment_file_path = os.path.join(sentiment_dir, ticker + '.csv')

    line_number = 0
    sentiments_ordered_by_date = []
    with open(sentiment_file_path, 'r') as sentiment_file:
        for line in sentiment_file:
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

            record = (ticker, filing_date, polarity, subjectivity)
            sentiments_ordered_by_date.append(record)

    return sentiments_ordered_by_date


def _get_10k_sentiment_for_ticker(ticker):
    return _get_filing_sentiment_for_ticker(ticker, '10')


def _get_8k_sentiment_for_ticker(ticker):
    return _get_filing_sentiment_for_ticker(ticker, '8')


def _get_min(sentiment_10k_list, sentiment_8k_list, sentiment_10k_index, sentiment_8k_index):
    if sentiment_10k_index == len(sentiment_10k_list):
        return sentiment_8k_list, sentiment_8k_index
    if sentiment_8k_index == len(sentiment_8k_list):
        return sentiment_10k_list, sentiment_10k_index

    if sentiment_8k_list[sentiment_8k_index][1] <= sentiment_10k_list[sentiment_10k_index][1]:
        return sentiment_8k_list, sentiment_8k_index
    else:
        return sentiment_10k_list, sentiment_10k_index


def _price_to_trend_bit(price_at_horizon, base_price):
    price_trend_percentage = price_at_horizon / base_price
    return 1 if price_trend_percentage >= 1 else 0


def _append_8k_row_for_ticker(data_frame_csv_file,
                              sentiment_8k_row,
                              eps_list_for_ticker,
                              open_prices_dict,
                              specialized_sentiment_8k_by_date,
                              departures_per_date,
                              averages_8k):
    filing_date = sentiment_8k_row[1]
    try:
        eps_surprise_percentage, days_since_eps = _get_relevant_eps_data(filing_date, eps_list_for_ticker)
    except ValueError:
        return
    specialized_sentiment = specialized_sentiment_8k_by_date[filing_date]
    try:
        ceo_departure, cfo_departure = departures_per_date[filing_date]
    except KeyError:
        ceo_departure = 0
        cfo_departure = 0
    base_price = _find_base_price(open_prices_dict, filing_date)
    try:
        price_trends = [_price_to_trend_bit(_find_price_at_horizon(open_prices_dict, filing_date, horizon), base_price)
                        for horizon in horizons]
    except TypeError:
        return

    record = (sentiment_8k_row[0],  # ticker
              str(sentiment_8k_row[1]),  # date

              '1',  # 1 for 8k, 0 for 10k

              str(eps_surprise_percentage),  # eps_percentage
              str(days_since_eps),  # days since eps

              str(sentiment_8k_row[2]),  # polarity
              str(sentiment_8k_row[3]),  # subjectivity

              str(specialized_sentiment[2]),  # Negative
              str(specialized_sentiment[3]),  # Positive
              str(specialized_sentiment[4]),  # Uncertainty
              str(specialized_sentiment[5]),  # Litigious
              str(specialized_sentiment[6]),  # Constraining
              str(specialized_sentiment[7]),  # Superfluous
              str(specialized_sentiment[8]),  # Interesting
              str(specialized_sentiment[9]),  # Modal
              str(specialized_sentiment[10]),  # WordCount

              '0',  # cosine

              str(ceo_departure),  # ceo_departure
              str(cfo_departure),  # cfo_departure

              str(averages_8k[0]),  # Avg_Polarity
              str(averages_8k[1]),  # Avg_Subjectivity
              str(averages_8k[2]),  # Avg_Negative
              str(averages_8k[3]),  # Avg_Positive
              str(averages_8k[4]),  # Avg_Litigious
              str(averages_8k[5]),  # Avg_Constraining
              str(averages_8k[6]),  # Avg_Superfluous
              str(averages_8k[7]),  # Avg_Modal
              str(averages_8k[8]),  # Avg_WordCount
              '0',  # Avg_Cosine
              str(averages_8k[9]),  # Avg_CEO_departure
              str(averages_8k[10]),  # Avg_CFO_departure

              str(price_trends[0]),  # 5 days
              str(price_trends[1]),  # 30 days
              str(price_trends[2]),  # 90 days
              str(price_trends[3]),  # 180 days
              str(price_trends[4]),  # 360 days
              str(price_trends[5]),  # 720 days
              str(price_trends[6]),  # 1080 days
              )

    data_frame_csv_file.write(','.join(record) + '\n')

    averages_8k = (
        (sentiment_8k_row[2] / 2) + (averages_8k[0] / 2),  # Avg_Polarity
        (sentiment_8k_row[3] / 2) + (averages_8k[1] / 2),  # Avg_Subjectivity
        (specialized_sentiment[2] / 2) + (averages_8k[2] / 2),  # Avg_Negative
        (specialized_sentiment[3] / 2) + (averages_8k[3] / 2),  # Avg_Positive
        (specialized_sentiment[4] / 2) + (averages_8k[4] / 2),  # Avg_Litigious
        (specialized_sentiment[5] / 2) + (averages_8k[5] / 2),  # Avg_Constraining
        (specialized_sentiment[6] / 2) + (averages_8k[6] / 2),  # Avg_Superfluous
        (specialized_sentiment[7] / 2) + (averages_8k[7] / 2),  # Avg_Modal
        (specialized_sentiment[8] / 2) + (averages_8k[8] / 2),  # Avg_WordCount
        (ceo_departure / 2) + (averages_8k[9] / 2),  # Avg_CEO_Departure
        (cfo_departure / 2) + (averages_8k[10] / 2)  # Avg_CFO_Departure
    )

    return averages_8k


def _append_10k_row_for_ticker(data_frame_csv_file,
                               sentiment_10k_row,
                               eps_list_for_ticker,
                               open_prices_dict,
                               specialized_sentiment_10k_by_date,
                               cosine_indices,
                               averages_10k):
    filing_date = sentiment_10k_row[1]
    try:
        eps_surprise_percentage, days_since_eps = _get_relevant_eps_data(filing_date, eps_list_for_ticker)
    except ValueError:
        return
    specialized_sentiment = specialized_sentiment_10k_by_date[filing_date]
    base_price = _find_base_price(open_prices_dict, filing_date)
    try:
        price_trends = [_price_to_trend_bit(_find_price_at_horizon(open_prices_dict, filing_date, horizon), base_price)
                        for horizon in horizons]
    except TypeError:
        return

    record = (sentiment_10k_row[0],  # ticker
              str(sentiment_10k_row[1]),  # date

              '0',  # 1 for 8k, 0 for 10k

              str(eps_surprise_percentage),  # eps_percentage
              str(days_since_eps),  # days since eps

              str(sentiment_10k_row[2]),  # polarity
              str(sentiment_10k_row[3]),  # subjectivity

              str(specialized_sentiment[2]),  # Negative
              str(specialized_sentiment[3]),  # Positive
              str(specialized_sentiment[4]),  # Uncertainty
              str(specialized_sentiment[5]),  # Litigious
              str(specialized_sentiment[6]),  # Constraining
              str(specialized_sentiment[7]),  # Superfluous
              str(specialized_sentiment[8]),  # Interesting
              str(specialized_sentiment[9]),  # Modal
              str(specialized_sentiment[10]),  # WordCount

              str(cosine_indices[filing_date]),  # cosine

              '0',  # ceo_departure
              '0',  # cfo_departure

              str(averages_10k[0]),  # Avg_Polarity
              str(averages_10k[1]),  # Avg_Subjectivity
              str(averages_10k[2]),  # Avg_Negative
              str(averages_10k[3]),  # Avg_Positive
              str(averages_10k[4]),  # Avg_Litigious
              str(averages_10k[5]),  # Avg_Constraining
              str(averages_10k[6]),  # Avg_Superfluous
              str(averages_10k[7]),  # Avg_Modal
              str(averages_10k[8]),  # Avg_WordCount
              str(averages_10k[9]),  # Avg_Cosine
              '0',  # Avg_CEO_Departure
              '0',  # Avg_CFO_Departure

              str(price_trends[0]),  # 5 days
              str(price_trends[1]),  # 30 days
              str(price_trends[2]),  # 90 days
              str(price_trends[3]),  # 180 days
              str(price_trends[4]),  # 360 days
              str(price_trends[5]),  # 720 days
              str(price_trends[6]),  # 1080 days
              )

    data_frame_csv_file.write(','.join(record) + '\n')

    averages_10k = (
        (sentiment_10k_row[2] / 2) + (averages_10k[0] / 2),  # Avg_Polarity
        (sentiment_10k_row[3] / 2) + (averages_10k[1] / 2),  # Avg_Subjectivity
        (specialized_sentiment[2] / 2) + (averages_10k[2] / 2),  # Avg_Negative
        (specialized_sentiment[3] / 2) + (averages_10k[3] / 2),  # Avg_Positive
        (specialized_sentiment[4] / 2) + (averages_10k[4] / 2),  # Avg_Litigious
        (specialized_sentiment[5] / 2) + (averages_10k[5] / 2),  # Avg_Constraining
        (specialized_sentiment[6] / 2) + (averages_10k[6] / 2),  # Avg_Superfluous
        (specialized_sentiment[7] / 2) + (averages_10k[7] / 2),  # Avg_Modal
        (specialized_sentiment[8] / 2) + (averages_10k[8] / 2),  # Avg_WordCount
        (cosine_indices[filing_date] / 2) + (averages_10k[8] / 2)  # Avg_Cosine
    )

    return averages_10k


def _append_rows_for_ticker(data_frame_csv_file,
                            ticker,
                            specialized_sentiment_10k_per_ticker,
                            specialized_sentiment_8k_per_ticker,
                            cosine_indices_per_ticker,
                            departures_per_ticker):
    # order the following by date:
    # get all 10k sentiments
    # get all 10k specialized sentiments
    # get all 10k cosine

    # order the following by date:
    # get all 8k sentiments
    # get all 8k specialized sentiments
    # get all 8k departures

    # get trend for each of the horizons from the filing date of each filing
    # get eps for each of the filing dates

    try:
        sentiment_10k_list = _get_10k_sentiment_for_ticker(ticker)
    except FileNotFoundError:
        sentiment_10k_list = []
    sentiment_8k_list = _get_8k_sentiment_for_ticker(ticker)
    eps_list_for_ticker = _get_eps_list_from_file(ticker)
    open_prices_dict, close_prices_dict = _get_prices_for_ticker(ticker)

    sentiment_10k_index = 0
    sentiment_8k_index = 0

    averages_10k = (
        0,  # Avg_Polarity
        0,  # Avg_Subjectivity
        0,  # Avg_Negative
        0,  # Avg_Positive
        0,  # Avg_Litigious
        0,  # Avg_Constraining
        0,  # Avg_Superfluous
        0,  # Avg_Modal
        0,  # Avg_WordCount
        0  # Avg_Cosine
    )

    averages_8k = (
        0,  # Avg_Polarity
        0,  # Avg_Subjectivity
        0,  # Avg_Negative
        0,  # Avg_Positive
        0,  # Avg_Litigious
        0,  # Avg_Constraining
        0,  # Avg_Superfluous
        0,  # Avg_Modal
        0,  # Avg_WordCount
        0,  # Avg_CEO_Departure
        0  # Avg_CFO_Departure
    )

    while ((sentiment_10k_index < len(sentiment_10k_list) and
            sentiment_10k_list[sentiment_10k_index][1] < datetime.date(2010, 1, 1))
           and
           (sentiment_8k_index < len(sentiment_8k_list) and
            sentiment_8k_list[sentiment_8k_index][1] < datetime.date(2010, 1, 1))):

        min_list, min_index = _get_min(sentiment_10k_list, sentiment_8k_list, sentiment_10k_index, sentiment_8k_index)
        if min_list == sentiment_8k_list:
            if specialized_sentiment_8k_per_ticker.get(ticker) is not None:
                retval = _append_8k_row_for_ticker(data_frame_csv_file,
                                                   sentiment_8k_list[sentiment_8k_index],
                                                   eps_list_for_ticker,
                                                   open_prices_dict,
                                                   specialized_sentiment_8k_per_ticker[ticker],
                                                   departures_per_ticker[ticker],
                                                   averages_8k)
                if retval is not None:
                    averages_8k = retval
            sentiment_8k_index += 1
        else:
            if specialized_sentiment_10k_per_ticker.get(ticker) is not None:
                retval = _append_10k_row_for_ticker(data_frame_csv_file,
                                                    sentiment_10k_list[sentiment_10k_index],
                                                    eps_list_for_ticker,
                                                    open_prices_dict,
                                                    specialized_sentiment_10k_per_ticker[ticker],
                                                    cosine_indices_per_ticker[ticker],
                                                    averages_10k)
                if retval is not None:
                    averages_10k = retval
            sentiment_10k_index += 1


def create_data_frame_csv_file(data_frame_csv_file_path):
    specialized_sentiment_10k_per_ticker = _get_10k_specialized_sentiment()
    specialized_sentiment_8k_per_ticker = _get_8k_specialized_sentiment()
    cosine_indices_per_ticker = _get_cosine_index_for_10k()
    departures_per_ticker = management_departure_indexer.departures_from_file()

    with open(data_frame_csv_file_path, 'w') as data_frame_csv_file:
        data_frame_csv_file.write(','.join(columns) + '\n')

        for ticker in TICKERS:
            print('Creating data frame for:' + ticker + '...')
            _append_rows_for_ticker(data_frame_csv_file,
                                    ticker,
                                    specialized_sentiment_10k_per_ticker,
                                    specialized_sentiment_8k_per_ticker,
                                    cosine_indices_per_ticker,
                                    departures_per_ticker
                                    )
            print('Done.')


if __name__ == '__main__':
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_frame_dir = os.path.join(current_file_dir_path, '..', 'data', 'Data_Frames')
    data_frame_csv_file_path = os.path.join(data_frame_dir, 'relations_cosine_specialized_sentiment_1080.csv')

    if not os.path.exists(data_frame_dir):
        os.makedirs(data_frame_dir)

    create_data_frame_csv_file(data_frame_csv_file_path)
