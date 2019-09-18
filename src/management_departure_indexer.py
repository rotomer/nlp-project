import datetime
import os

ceo_tokens = ['chief executive officer',
              'ceo']
cfo_tokens = ['chief financial officer',
              'cfo']
departure_tokens = ['depart',
                    'departure',
                    'leave',
                    'leaving',
                    'retire',
                    'retirement',
                    'resign',
                    'resignation',
                    'terminate',
                    'termination',
                    'cease',
                    'step down']


def _check_segment(match_targets, segment):
    return any(target in segment for target in match_targets)


def _check_ceo_hits(segments):
    return any(_check_segment(ceo_tokens, segment) for segment in segments)


def _check_cfo_hits(segments):
    return any(_check_segment(cfo_tokens, segment) for segment in segments)


def _check_departure_hits(segments):
    return any(_check_segment(departure_tokens, segment) for segment in segments)


def _departure_in_vicinity(hits, index):
    vicinity = 0

    if vicinity == 0:
        return hits[index]['Departures']
    else:
        for i in range(vicinity * -1, vicinity):
            if index + i < 0 or index + i >= len(hits):
                continue

            if hits[i]['Departures']:
                return True

        return False


def get_departures(relation_file_path):
    hits = []

    with open(relation_file_path, 'r') as relation_file:
        line_number = 0
        for line in relation_file:
            line_number += 1
            if line_number == 1:
                continue

            splits = line.lower().split(',')

            hits_dict = {
                'CEO': _check_ceo_hits(splits),
                'CFO': _check_cfo_hits(splits),
                'Departures': _check_departure_hits(splits)
            }
            hits.append(hits_dict)

    ceo_departures = 0
    cfo_departures = 0

    for i, hits_dict in enumerate(hits):
        if hits_dict['CEO']:
            if _departure_in_vicinity(hits, i):
                ceo_departures += 1
        if hits_dict['CFO']:
            if _departure_in_vicinity(hits, i):
                cfo_departures += 1

    return 1 if ceo_departures > 0 else 0, 1 if cfo_departures > 0 else 0


def index_departures():
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    relations_dir = os.path.join(current_file_dir_path, '..', 'data', 'Relations_8K')
    departures_dir = os.path.join(current_file_dir_path, '..', 'data', 'Departures_8K')

    if not os.path.exists(departures_dir):
        os.makedirs(departures_dir)

    departures_file_path = os.path.join(departures_dir, 'departures.csv')

    with open(departures_file_path, 'w') as departures_file:
        departures_file.write(','.join(['Ticker', 'FileName', 'CEO_Departure', 'CFO_Departure']) + '\n')

        for folder_name in os.listdir(relations_dir):
            folder_path = os.path.join(relations_dir, folder_name)
            for file_name in os.listdir(folder_path):
                relations_file_path = os.path.join(folder_path, file_name)
                print('Analyzing departures for: ' + relations_file_path + '...')
                ceo_departure, cfo_departure = get_departures(relations_file_path)
                departures_file.write(','.join([folder_name, file_name, str(ceo_departure), str(cfo_departure)]) + '\n')
                print('Done.')


def departures_from_file():
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    departures_file_path = os.path.join(current_file_dir_path, '..', 'data', 'Departures_8K', 'departures.csv')

    departures_per_ticker = {}
    line_number = 0
    last_ticker = ''
    ticker_departures_by_date = {}
    with open(departures_file_path, 'r') as departures_file:
        for line in departures_file:
            line_number = line_number + 1
            if line_number == 1:
                continue

            line = line.strip()
            if line == '':
                continue

            splits = line.split(',')

            ticker = splits[0]

            if last_ticker != ticker:
                if len(ticker_departures_by_date) > 0:
                    departures_per_ticker[last_ticker] = ticker_departures_by_date
                ticker_departures_by_date = {}
                last_ticker = ticker

            file_name = splits[1]
            filing_date_str = file_name.split('.')[0]
            filing_date = datetime.datetime.strptime(filing_date_str, "%Y-%m-%d").date()
            ceo_departure = int(splits[2])
            cfo_departure = int(splits[3])

            record = (ceo_departure, cfo_departure)
            ticker_departures_by_date[filing_date] = record

    return departures_per_ticker


if __name__ == '__main__':
    index_departures()