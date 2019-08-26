import re

import requests

DEFAULT_TICKER = 'aapl'
URL = 'http://www.sec.gov/cgi-bin/browse-edgar?CIK={}&Find=Search&owner=exclude&action=getcompany'
CIK_RE = re.compile(r'.*CIK=(\d{10}).*')


def fetch_cik_from_edgar(ticker):
    f = requests.get(URL.format(ticker), stream=True)
    results = CIK_RE.findall(f.text)
    if len(results):
        return str(results[0])

    return None


if __name__ == '__main__':
    print(fetch_cik_from_edgar(DEFAULT_TICKER))
