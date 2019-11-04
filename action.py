from bitfinex import WssClient, ClientV2, ClientV1
from UTILS.cointegration_analysis import estimate_long_run_short_run_relationships, \
    engle_granger_two_step_cointegration_test
from datetime import *
import pandas as pd
import numpy as np
from UTILS import FileIO

# Define the number of samples to be buffered, gamma and threshold values from analysis stage
instrument_1 = 'tBABUSD'
instrument_2 = 'tETHUSD'
buffer_count = 50
max_size = 500
gamma_value = 0.3939
threshold_value = 0.000025

# Read config
config_path  = 'CONFIG\config_finex.yml'
config_finex = FileIO.read_yaml(config_path)

key = config_finex['key']
secret = config_finex['secret']

# Do not touch this part.
j = 0
chan_id = None
ref = {}
cache = {}
data_combined = []
ask1_cached = 0
ask2_cached = 0
bid1_cached = 0
bid2_cached = 0
bid_total = 0
pos = 0

def my_handler(message):
    global j
    global chan_id
    global ask1_cached
    global ask2_cached
    global bid1_cached
    global bid2_cached
    global bid_total
    global pos

    if isinstance(message, dict):
        if message['event'] == 'subscribed':
            if message['channel'] == 'ticker':
                chan_id = message['chanId']
                ref[message['symbol']] = chan_id
                # print(ref)

    if isinstance(message, list):
        if message[1] != 'hb':
            symbol = list(ref.keys())[list(ref.values()).index(message[0])]
            cache[symbol] = [message[1][0], message[1][1], message[1][2], message[1][3]]

        if len(cache) >= 2:  # must be equal to the number of tickers subscribed.
            price_info = []
            time_ = datetime.utcnow()
            price_info.append(time_)
            for i in cache:
                price_info.append(cache[i][0])
                price_info.append(cache[i][1])
                price_info.append(cache[i][2])
                price_info.append(cache[i][3])
            if len(data_combined) <= max_size:
                print('Samples recorded:', len(data_combined))
            if len(data_combined) > max_size:
                del data_combined[0]
            data_combined.append(price_info)

        j += 1

    if j != 0 and j % 5 == 0:
        print(j)
    if j >= buffer_count:
        list_columns = []
        columns_temp = ['date']
        for k in cache:
            ticker = k[1:-3]
            columns_temp.append('BidPrice-' + ticker)
            columns_temp.append('BidVolume-' + ticker)
            columns_temp.append('AskPrice-' + ticker)
            columns_temp.append('AskVolume-' + ticker)
        list_columns.append(columns_temp)
        market_data_raw = pd.DataFrame(data_combined, columns=list_columns[0])
        market_data_raw.set_index('date', inplace=True)
        market_data_raw.columns = [market_data_raw.columns.str[-3:], market_data_raw.columns.str[:-4]]
        market_data = market_data_raw
        # print(market_data)
        # Show the First 5 Rows
        # print(market_data.head(5))

        # Show the Stocks
        stock_names = list(market_data.columns.get_level_values(0).unique())
        # print('The stocks available are', stock_names)

        # Calculate mid-prices of each stock and add them to the DataFrame
        for stock in stock_names:
            market_data[stock, 'MidPrice'] = (market_data[stock, 'BidPrice'] + market_data[stock, 'AskPrice']) / 2
            market_data = market_data.sort_index(axis=1)

        # print(market_data.head(5))

        # Obtain the statistical parameters for each and every pair
        data_analysis = {'Pairs': [],
                         'Constant': [],
                         'Gamma': [],
                         'Alpha': [],
                         'P-Value': []}

        data_zvalues = {}

        for stock1 in stock_names:
            for stock2 in stock_names:
                if stock1 != stock2:
                    if (stock2, stock1) in data_analysis['Pairs']:
                        continue

                    pairs = stock1, stock2
                    constant = estimate_long_run_short_run_relationships(np.log(
                        market_data[stock1, 'MidPrice']), np.log(market_data[stock2, 'MidPrice']))[0]
                    gamma = estimate_long_run_short_run_relationships(np.log(
                        market_data[stock1, 'MidPrice']), np.log(market_data[stock2, 'MidPrice']))[1]
                    alpha = estimate_long_run_short_run_relationships(np.log(
                        market_data[stock1, 'MidPrice']), np.log(market_data[stock2, 'MidPrice']))[2]
                    pvalue = engle_granger_two_step_cointegration_test(np.log(
                        market_data[stock1, 'MidPrice']), np.log(market_data[stock2, 'MidPrice']))[1]
                    zvalue = estimate_long_run_short_run_relationships(np.log(
                        market_data[stock1, 'MidPrice']), np.log(market_data[stock2, 'MidPrice']))[3]

                    data_analysis['Pairs'].append(pairs)
                    data_analysis['Constant'].append(constant)
                    data_analysis['Gamma'].append(gamma)
                    data_analysis['Alpha'].append(alpha)
                    data_analysis['P-Value'].append(pvalue)

                    data_zvalues[pairs] = zvalue

        data_analysis = round(pd.DataFrame(data_analysis), 4).set_index('Pairs')

        # Selecting tradable pairs where P-Value < 0.01 and create a seperate DataFrame containing these pairs
        tradable_pairs_analysis = data_analysis[data_analysis['P-Value'] < 0.01].sort_values('P-Value')

        # Get all the tradable stock pairs into a list
        stock_pairs = [[instrument_1[1:-3], instrument_2[1:-3]]]

        # Create a list of unique tradable stocks
        list_stock1 = [stock[0] for stock in stock_pairs]
        list_stock2 = [stock[1] for stock in stock_pairs]

        for stock in list_stock2:
            list_stock1.append(stock)

        unique_stock_list = list(set(list_stock1))

        # Create a new DataFrame containing all market information for the tradable pairs
        tradable_pairs_data = market_data[unique_stock_list]

        # Create a new column within the earlier defined DataFrame with Z-Values of all stock pairs
        for pair in stock_pairs:
            stock1 = pair[0]
            stock2 = pair[1]
            # print('check:', data_zvalues)
            tradable_pairs_data[stock1 + stock2, 'Z-Value'] = data_zvalues[stock1, stock2]

        # Selection of the final pairs for this trading strategy
        stock_pairs_final = [[instrument_1[1:-3], instrument_2[1:-3]]]

        positions_strategy_1 = {}
        limit = 100

        for pair in stock_pairs_final:
            # print(pair)
            stock1 = pair[0]
            stock2 = pair[1]

            gamma = gamma_value

            threshold = threshold_value

            current_position_stock1 = 0
            current_position_stock2 = 0

            positions_strategy_1[stock1] = []

            for _, data_at_time in tradable_pairs_data.iterrows():
                # print(data_at_time)
                BidPrice_Stock1 = data_at_time[stock1, 'BidPrice']
                AskPrice_Stock1 = data_at_time[stock1, 'AskPrice']
                BidPrice_Stock2 = data_at_time[stock2, 'BidPrice']
                AskPrice_Stock2 = data_at_time[stock2, 'AskPrice']

                BidVolume_Stock1 = data_at_time[stock1, 'BidVolume']
                AskVolume_Stock1 = data_at_time[stock1, 'AskVolume']
                BidVolume_Stock2 = data_at_time[stock2, 'BidVolume']
                AskVolume_Stock2 = data_at_time[stock2, 'AskVolume']

                zvalue = data_at_time[stock1 + stock2, 'Z-Value']

                if zvalue >= threshold:
                    hedge_ratio = gamma * (BidPrice_Stock1 / AskPrice_Stock2)

                    if hedge_ratio >= 1:

                        max_order_stock1 = current_position_stock1 + limit
                        max_order_stock2 = max_order_stock1 / hedge_ratio

                        trade = np.floor(
                            min((BidVolume_Stock1 / hedge_ratio), AskVolume_Stock2, max_order_stock1, max_order_stock2))

                        positions_strategy_1[stock1].append((- trade * hedge_ratio) + current_position_stock1)

                        current_position_stock1 = ((- trade * hedge_ratio) + current_position_stock1)

                    elif hedge_ratio < 1:

                        max_order_stock1 = current_position_stock1 + limit
                        max_order_stock2 = max_order_stock1 * hedge_ratio

                        trade = np.floor(
                            min((BidVolume_Stock1 * hedge_ratio), AskVolume_Stock2, max_order_stock1, max_order_stock2))

                        positions_strategy_1[stock1].append((- trade / hedge_ratio) + current_position_stock1)

                        current_position_stock1 = ((- trade / hedge_ratio) + current_position_stock1)

                elif zvalue <= -threshold:
                    hedge_ratio = gamma * (AskPrice_Stock1 / BidPrice_Stock2)

                    if hedge_ratio >= 1:

                        max_order_stock1 = abs(current_position_stock1 - limit)
                        max_order_stock2 = max_order_stock1 / hedge_ratio

                        trade = np.floor(
                            min((AskVolume_Stock1 / hedge_ratio), BidVolume_Stock2, max_order_stock1, max_order_stock2))

                        positions_strategy_1[stock1].append((+ trade * hedge_ratio) + current_position_stock1)

                        current_position_stock1 = (+ trade * hedge_ratio) + current_position_stock1

                    elif hedge_ratio < 1:

                        max_order_stock1 = abs(current_position_stock1 - limit)
                        max_order_stock2 = max_order_stock1 * hedge_ratio

                        trade = np.floor(
                            min((AskVolume_Stock1 * hedge_ratio), BidVolume_Stock2, max_order_stock1, max_order_stock2))

                        positions_strategy_1[stock1].append((+ trade / hedge_ratio) + current_position_stock1)

                        current_position_stock1 = (+ trade / hedge_ratio) + current_position_stock1

                else:

                    positions_strategy_1[stock1].append(current_position_stock1)

            if hedge_ratio >= 1:
                positions_strategy_1[stock2] = positions_strategy_1[stock1] / hedge_ratio * -1

            elif hedge_ratio < 1:
                positions_strategy_1[stock2] = positions_strategy_1[stock1] / (1 / hedge_ratio) * -1

        # print(positions_strategy_1[instrument_1[1:-3]][len(positions_strategy_1[instrument_1[1:-3]])-1])
        # print(positions_strategy_1[instrument_2[1:-3]][len(positions_strategy_1[instrument_2[1:-3]])-1])

        if positions_strategy_1[instrument_1[1:-3]][len(positions_strategy_1[instrument_1[1:-3]])-1] < 0:
            if positions_strategy_1[instrument_2[1:-3]][len(positions_strategy_1[instrument_2[1:-3]])-1] > 0:
                print(instrument_1, ': SELL', BidPrice_Stock1, '/ ', instrument_2, ': BUY', BidPrice_Stock2)
                if pos == 0:
                    pos = -1
                    bid1_cached = BidPrice_Stock1
                    ask2_cached = AskPrice_Stock2
                if pos > 0:
                    delta_1 = BidPrice_Stock1 - ask1_cached
                    delta_2 = bid2_cached - AskPrice_Stock2
                    bid_total += (delta_1 + delta_2)
                    print('PnL cum:', bid_total)
                    bid1_cached = BidPrice_Stock1
                    ask2_cached = AskPrice_Stock2
                    pos = -1
        if positions_strategy_1[instrument_1[1:-3]][len(positions_strategy_1[instrument_1[1:-3]])-1] > 0:
            if positions_strategy_1[instrument_2[1:-3]][len(positions_strategy_1[instrument_2[1:-3]])-1] < 0:
                print(instrument_1, ': BUY', BidPrice_Stock1, '/ ', instrument_2, ': SELL', BidPrice_Stock2)
                if pos == 0:
                    pos = 1
                    ask1_cached = AskPrice_Stock1
                    bid2_cached = BidPrice_Stock2
                if pos < 0:
                    bid1_delta = bid1_cached - AskPrice_Stock1
                    bid2_delta = BidPrice_Stock2 - ask2_cached
                    bid_total += (bid1_delta + bid2_delta)
                    print('PnL cum:', bid_total)
                    ask1_cached = AskPrice_Stock1
                    bid2_cached = BidPrice_Stock2
                    pos = 1


# Just add and remove tickers here.

my_client = WssClient(key, secret)
my_client.subscribe_to_ticker(
    symbol=instrument_1,
    callback=my_handler
)
my_client.subscribe_to_ticker(
    symbol=instrument_2,
    callback=my_handler
)

try:
    print('It began in Africa...')
    my_client.start()

except KeyboardInterrupt:
    my_client.stop()
    print('  ---This is the end !---')
