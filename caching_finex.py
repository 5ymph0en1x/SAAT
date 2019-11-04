from bitfinex import WssClient, ClientV2, ClientV1
from datetime import *
import pandas as pd
from UTILS import FileIO

# Define the number of samples to be captured and the resulting filename
samples_count = 5000
results_filename = "sample_.csv"

# Read config
config_path = 'CONFIG\config_finex.yml'
config_finex = FileIO.read_yaml(config_path)

key = config_finex['key']
secret = config_finex['secret']

# Do not touch this part.
j = 0
chan_id = None
ref = {}
cache = {}
data_combined = []


def my_handler(message):
    global j
    global chan_id

    if isinstance(message, dict):
        if message['event'] == 'subscribed':
            if message['channel'] == 'ticker':
                chan_id = message['chanId']
                ref[message['symbol']] = chan_id
                print(ref)

    if isinstance(message, list):
        if message[1] != 'hb':
            symbol = list(ref.keys())[list(ref.values()).index(message[0])]
            cache[symbol] = [message[1][0], message[1][1], message[1][2], message[1][3]]

        if len(cache) >= 8:  # must be equal to the number of tickers subscribed.
            price_info = []
            time_ = datetime.utcnow()
            price_info.append(time_)
            for i in cache:
                price_info.append(cache[i][0])
                price_info.append(cache[i][1])
                price_info.append(cache[i][2])
                price_info.append(cache[i][3])
            data_combined.append(price_info)

        j += 1

    if j != 0 and j % 50 == 0:
        print(j)
    if j >= samples_count:
        list_columns = []
        columns_temp = ['']
        for k in cache:
            ticker = k[1:-3]
            columns_temp.append('BidPrice-' + ticker)
            columns_temp.append('BidVolume-' + ticker)
            columns_temp.append('AskPrice-' + ticker)
            columns_temp.append('AskVolume-' + ticker)
        list_columns.append(columns_temp)
        print(list_columns[0])
        to_save = pd.DataFrame(data_combined, columns=list_columns[0])
        to_save.to_csv(path_or_buf=results_filename, index=False)
        my_client.stop()


# Just add and remove tickers here.

my_client = WssClient(key, secret)
my_client.subscribe_to_ticker(
    symbol="tBABUSD",
    callback=my_handler
)
my_client.subscribe_to_ticker(
    symbol="tBSVUSD",
    callback=my_handler
)
my_client.subscribe_to_ticker(
    symbol="tBTCUSD",
    callback=my_handler
)
my_client.subscribe_to_ticker(
    symbol="tEOSUSD",
    callback=my_handler
)
my_client.subscribe_to_ticker(
    symbol="tETHUSD",
    callback=my_handler
)
my_client.subscribe_to_ticker(
    symbol="tLTCUSD",
    callback=my_handler
)
my_client.subscribe_to_ticker(
    symbol="tNEOUSD",
    callback=my_handler
)
my_client.subscribe_to_ticker(
    symbol="tXRPUSD",
    callback=my_handler
)

try:
    print('It began in Africa...')
    my_client.start()

except KeyboardInterrupt:
    my_client.stop()
    print('  ---This is the end !---')
