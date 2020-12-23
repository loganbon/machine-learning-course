import os
import requests
import pandas as pd

def setKey(key):
    if type(key) == str:
        os.environ['ALPHA_KEY'] = key
    else:
        print('Environment Variables must be of type str.')

def getData(symbols):    
    api_key = os.getenv('ALPHA_KEY')

    base_url = 'https://www.alphavantage.co/query?'

    for i in range(len(symbols)):
        params = {'function': 'TIME_SERIES_DAILY',
            'symbol': symbols[i],
            'outputsize': 'full',
            'datatype':'json',
            'apikey': api_key}

        response = requests.get(base_url, params=params)
   
        data = pd.DataFrame.from_dict(response.json()['Time Series (Daily)'])
        data = data.loc['4. close'].to_frame(name=symbols[i])
        data['Date'] = data.index
        data.to_csv('data/{}.csv'.format(symbols[i]), header=True, index=False)