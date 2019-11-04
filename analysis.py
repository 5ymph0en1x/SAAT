from UTILS.cointegration_analysis import estimate_long_run_short_run_relationships, \
    engle_granger_two_step_cointegration_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Name of the previously collected data
results_filename = "sample_.csv"
analysis_type = 1  # 1- Partial / 2- Full

# Start of the analysis
def read_data(filename):
    '''
    This function reads the .csv stored at the 'filename' location and returns a DataFrame
    with two levels of column names. The first level column contains the Stock Name and the
    second contains the type of market data, e.g. bid/ask, price/volume.
    '''
    df = pd.read_csv(filename, index_col=0)
    df.columns = [df.columns.str[-3:], df.columns.str[:-4]]
    print(df)
    return df


market_data = read_data(results_filename)

# Show the First 5 Rows
print(market_data.head(5))

# Show the Stocks
stock_names = list(market_data.columns.get_level_values(0).unique())
print('The stocks available are', stock_names)

# Calculate mid-prices of each stock and add them to the DataFrame
for stock in stock_names:
    market_data[stock, 'MidPrice'] = (market_data[stock, 'BidPrice'] + market_data[stock, 'AskPrice']) / 2
    market_data = market_data.sort_index(axis=1)

print(market_data.head(5))


def mid_price_check(stock):
    '''
    Function that checks for different stocks if the MidPrice
    is correctly specified.
    '''
    plt.figure(figsize=(20, 5))
    plt.plot(market_data[stock, 'AskPrice'][:100])
    plt.plot(market_data[stock, 'MidPrice'][:100])
    plt.plot(market_data[stock, 'BidPrice'][:100])

    plt.xticks([])  # Timestamp is not Important
    plt.title('Ask, Bid and Mid Price Development of Stock ' + stock)
    plt.legend(["Ask Price", "Mid Price", "Bid Price"], loc='lower left')
    plt.show()


mid_price_check('BTC')

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

# Visualize the P-values
def plot_pvalues():
    """
    This function plots all obtained P-values.
    """
    plt.figure(figsize=(20, 5))
    plt.hist(data_analysis['P-Value'], bins=100)
    plt.xlabel('P-value')
    plt.ylabel('Number of observations')
    plt.title('All obtained P-values')
    plt.show()

plot_pvalues()

# Show Top 10 and Bottom 10
print(data_analysis.sort_values('P-Value')[:10])
print(data_analysis.sort_values('P-Value')[-10:])

# Selecting tradable pairs where P-Value < 0.01 and create a seperate DataFrame containing these pairs
tradable_pairs_analysis = data_analysis[data_analysis['P-Value'] < 0.01].sort_values('P-Value')

# Get all the tradable stock pairs into a list
stock_pairs = list(tradable_pairs_analysis.index.values.tolist())

# Show the Pairs
print(stock_pairs)

# Create a list of unique tradable stocks
list_stock1 = [stock[0] for stock in stock_pairs]
list_stock2 = [stock[1] for stock in stock_pairs]

for stock in list_stock2:
    list_stock1.append(stock)

unique_stock_list = list(set(list_stock1))

# Create a new DataFrame containing all market information for the tradable pairs
tradable_pairs_data = market_data[unique_stock_list]
print(tradable_pairs_data.head())


def Plot_Tradable_Z():
    """
    This function plots the z-values of all pairs based on
    the data_zvalues dataframe.
    """
    for pair in stock_pairs:
        zvalue = data_zvalues[pair]
        plt.figure(figsize=(20, 5))
        plt.title('Error-correction term stock pair {}'.format(pair))
        zvalue.plot()
        plt.xlabel('Time')
        plt.ylabel('Magnitude')

        xmin = 0
        xmax = len(zvalue)
        plt.hlines(0.005, xmin, xmax, 'g')  # Note 0.005 is randomly chosen
        plt.hlines(-0.005, xmin, xmax, 'r')  # Note -0.005 is randomly chosen

        plt.legend(['Z-Value', 'Positive Threshold', 'Negative Threshold'], loc='lower left')

        plt.show()


Plot_Tradable_Z()

# Select randomly chosen pair from the tradable stock and visualize bid and ask prices, bid and ask volumes, and the z-values
import random

# Choose random stock
random_pair = random.choice(stock_pairs)


# Create a plot showing the bid and ask prices of a randomly chosen stock
def Plot_RandomPair_BidAskPrices():
    """
    This function plots the bid and ask price of a randomly chosen tradable pair.
    """
    plt.figure(figsize=(20, 5))
    plt.title('Bid and ask prices of stock pair {} and {}'.format(random_pair[0], random_pair[1]))

    plt.plot(tradable_pairs_data[random_pair[0], 'AskPrice'].iloc[:100], 'r')
    plt.plot(tradable_pairs_data[random_pair[0], 'BidPrice'].iloc[:100], 'm')
    plt.xlabel('Time')
    plt.ylabel('Price stock {}'.format(random_pair[0]))
    plt.legend(loc='lower left')

    plt.twinx()
    plt.plot(tradable_pairs_data[random_pair[1], 'AskPrice'].iloc[:100])
    plt.plot(tradable_pairs_data[random_pair[1], 'BidPrice'].iloc[:100])
    plt.xticks([])
    plt.ylabel('Price stock {}'.format(random_pair[1]))
    plt.legend(loc='upper right')

    plt.show()


# Plot_RandomPair_BidAskPrices()


# Create a plot showing the bid and ask volumes of a randomly chosen stock
def Plot_RandomPair_BidAskVolumes():  # Plot not really clarifying, maybe other kind of plot?
    """
    This function plots the bid and ask volumes of a randomly chosen tradable pair.
    """
    plt.figure(figsize=(20, 5))
    plt.title('Bid and ask volumes of stock pair {} and {}'.format(random_pair[0], random_pair[1]))

    plt.plot(tradable_pairs_data[random_pair[0], 'AskVolume'].iloc[:100], 'r')
    plt.plot(tradable_pairs_data[random_pair[0], 'BidVolume'].iloc[:100], 'm')
    plt.xlabel('Time')
    plt.ylabel('Volume stock {}'.format(random_pair[0]))
    plt.legend(loc='lower left')

    plt.twinx()
    plt.plot(tradable_pairs_data[random_pair[1], 'AskVolume'].iloc[:100])
    plt.plot(tradable_pairs_data[random_pair[1], 'BidVolume'].iloc[:100])
    plt.xticks([])
    plt.ylabel('Volume stock {}'.format(random_pair[1]))
    plt.legend(loc='upper right')

    plt.show()


# Plot_RandomPair_BidAskVolumes()

# Create a Dataframe containing information about the error-correction term of each pair
data_error_correction_term = {'Pair': [],
                              'CountZeroCrossings': [],
                              'TradingPeriod': [],
                              'LongRunMean': [],
                              'Std': []}

for pair in stock_pairs:
    zvalue = data_zvalues[pair]
    my_array = np.array(zvalue)
    count = ((my_array[:-1] * my_array[1:]) < 0).sum()
    trading_period = 1 / count
    long_run_mean = zvalue.mean()
    std = zvalue.std()

    data_error_correction_term['Pair'].append(pair)
    data_error_correction_term['CountZeroCrossings'].append(count)
    data_error_correction_term['TradingPeriod'].append(trading_period)
    data_error_correction_term['LongRunMean'].append(round(long_run_mean, 4))
    data_error_correction_term['Std'].append(round(std, 4))

data_error_correction_term = pd.DataFrame(data_error_correction_term).set_index('Pair')

print(data_error_correction_term)

# Create a new column within the earlier defined DataFrame with Z-Values of all stock pairs
for pair in stock_pairs:
    stock1 = pair[0]
    stock2 = pair[1]

    tradable_pairs_data[stock1 + stock2, 'Z-Value'] = data_zvalues[stock1, stock2]

# Create a Dictionary that saves all Gamma values of each pair
gamma_dictionary = {}

for pair, value in tradable_pairs_analysis.iterrows():
    gamma_dictionary[pair] = value['Gamma']

print('Gamma:', gamma_dictionary)

# Create a Dictionary that saves all Standard Deviation values of each pair
std_dictionary = {}

for pair, value in data_error_correction_term.iterrows():
    std_dictionary[pair] = value['Std']

print('Deviation:', std_dictionary)

positions = {}
limit = 100

for pair in stock_pairs:
    stock1 = pair[0]
    stock2 = pair[1]

    gamma = gamma_dictionary[stock1, stock2]

    for i in np.linspace(0.05, 1.0, 10):
        threshold = float(i * std_dictionary[stock1, stock2])

        current_position_stock1 = 0
        current_position_stock2 = 0

        column_name_stock1 = stock1 + ' Pos - Thres: ' + str(threshold)

        # print(tradable_pairs_data[stock1, 'BidVolume'].iloc[0])
        BidPrice_Stock1 = tradable_pairs_data[stock1, 'BidVolume'].iloc[0]
        AskPrice_Stock1 = tradable_pairs_data[stock1, 'AskVolume'].iloc[0]
        BidPrice_Stock2 = tradable_pairs_data[stock2, 'BidVolume'].iloc[0]
        AskPrice_Stock2 = tradable_pairs_data[stock1, 'AskVolume'].iloc[0]

        positions[column_name_stock1] = []

        for time, data_at_time in tradable_pairs_data.iterrows():

            BidVolume_Stock1 = data_at_time[stock1, 'BidVolume']
            AskVolume_Stock1 = data_at_time[stock1, 'AskVolume']
            BidVolume_Stock2 = data_at_time[stock2, 'BidVolume']
            AskVolume_Stock2 = data_at_time[stock2, 'AskVolume']

            zvalue = float(data_at_time[stock1 + stock2, 'Z-Value'])

            # If the zvalues of (BB,DD) are high the spread diverges, i.e. sell BB (=stock1=y) and buy DD (=stock2=x)
            if zvalue >= threshold:
                hedge_ratio = float(gamma * (BidPrice_Stock1 / AskPrice_Stock2))

                if hedge_ratio >= 1:

                    max_order_stock1 = current_position_stock1 + limit
                    max_order_stock2 = max_order_stock1 / hedge_ratio

                    trade = np.floor(
                        min((BidVolume_Stock1 / hedge_ratio), AskVolume_Stock2, max_order_stock1, max_order_stock2))

                    positions[column_name_stock1].append((- trade * hedge_ratio) + current_position_stock1)

                    current_position_stock1 = ((- trade * hedge_ratio) + current_position_stock1)

                elif hedge_ratio < 1:

                    max_order_stock1 = current_position_stock1 + limit
                    max_order_stock2 = max_order_stock1 * hedge_ratio

                    trade = np.floor(
                        min((BidVolume_Stock1 * hedge_ratio), AskVolume_Stock2, max_order_stock1, max_order_stock2))

                    positions[column_name_stock1].append((- trade / hedge_ratio) + current_position_stock1)

                    current_position_stock1 = ((- trade / hedge_ratio) + current_position_stock1)

            elif zvalue <= -threshold:
                hedge_ratio = float(gamma * (AskPrice_Stock1 / BidPrice_Stock2))

                if hedge_ratio >= 1:

                    max_order_stock1 = abs(current_position_stock1 - limit)
                    max_order_stock2 = max_order_stock1 / hedge_ratio

                    trade = np.floor(
                        min((AskVolume_Stock1 / hedge_ratio), BidVolume_Stock2, max_order_stock1, max_order_stock2))

                    positions[column_name_stock1].append((+ trade * hedge_ratio) + current_position_stock1)

                    current_position_stock1 = (+ trade * hedge_ratio) + current_position_stock1

                elif hedge_ratio < 1:

                    max_order_stock1 = abs(current_position_stock1 - limit)
                    max_order_stock2 = max_order_stock1 * hedge_ratio

                    trade = np.floor(
                        min((AskVolume_Stock1 * hedge_ratio), BidVolume_Stock2, max_order_stock1, max_order_stock2))

                    positions[column_name_stock1].append((+ trade / hedge_ratio) + current_position_stock1)

                    current_position_stock1 = (+ trade / hedge_ratio) + current_position_stock1

                BidPrice_Stock1 = data_at_time[stock1, 'BidPrice']
                AskPrice_Stock1 = data_at_time[stock1, 'AskPrice']
                BidPrice_Stock2 = data_at_time[stock2, 'BidPrice']
                AskPrice_Stock2 = data_at_time[stock2, 'AskPrice']

            else:
                positions[column_name_stock1].append(current_position_stock1)

        column_name_stock2 = stock2 + ' Pos - Thres: ' + str(threshold)
        print(column_name_stock2)

        if hedge_ratio >= 1:
            # print(positions[column_name_stock1])
            positions[column_name_stock1] = np.array(positions[column_name_stock1])
            positions[column_name_stock2] = positions[column_name_stock1] / hedge_ratio * -1

        elif hedge_ratio < 1:
            positions[column_name_stock1] = np.array(positions[column_name_stock1])
            positions[column_name_stock2] = positions[column_name_stock1] / (1 / hedge_ratio) * -1

# Create a seperate dataframe (to keep the original dataframe intact) with rounding
# Also insert the timestamp, as found in the tradeable_pairs_data DataFrame
positions_final = np.ceil(pd.DataFrame(positions))
positions_final['Timestamp'] = tradable_pairs_data.index
positions_final = positions_final.set_index('Timestamp')

# The difference between the positions
positions_diff = positions_final.diff()[1:]

# Positions_diff first rows
print(positions_diff.head())

# OPTIONAL to Excel to Save the Amount of Trades
# positions_diff[(positions_diff != 0)].count().to_excel('Thresholds.xlsx')

positions_diff[-1:] = -positions_final[-1:]

pnl_dataframe = pd.DataFrame()

for pair in stock_pairs:
    stock1 = pair[0]
    stock2 = pair[1]

    Stock1_AskPrice = tradable_pairs_data[stock1, 'AskPrice'][1:]
    Stock1_BidPrice = tradable_pairs_data[stock1, 'BidPrice'][1:]
    Stock2_AskPrice = tradable_pairs_data[stock2, 'AskPrice'][1:]
    Stock2_BidPrice = tradable_pairs_data[stock2, 'BidPrice'][1:]

    for i in np.linspace(0.05, 1.0, 10):
        threshold = i * std_dictionary[stock1, stock2]

        column_name_1 = stock1 + ' Pos - Thres: ' + str(threshold)
        column_name_2 = stock2 + ' Pos - Thres: ' + str(threshold)

        pnl_dataframe[stock1 + str(threshold)] = np.where(positions_diff[column_name_1] > 0,
                                                          positions_diff[column_name_1] * -Stock1_BidPrice, positions_diff[column_name_1] * -Stock1_AskPrice)
        pnl_dataframe[stock2 + str(threshold)] = np.where(positions_diff[column_name_2] > 0,
                                                          positions_diff[column_name_2] * -Stock2_BidPrice, positions_diff[column_name_2] * -Stock2_AskPrice)

print(pnl_dataframe.head())

# Create Columns for the pnl_threshold dataframe
pairs = []
thresholds = []

for pair in stock_pairs:
    stock1 = pair[0]
    stock2 = pair[1]

    for i in np.linspace(0.05, 1.0, 10):
        threshold = i * std_dictionary[stock1, stock2]
        pair = stock1, stock2
        pairs.append(pair)
        thresholds.append(threshold)

# Include columns and append PnLs
pnl_threshold = {'Pairs': pairs,
                 'Thresholds': thresholds,
                 'PnLs': []}

for pair in stock_pairs:
    stock1 = pair[0]
    stock2 = pair[1]

    for i in np.linspace(0.05, 1.0, 10):
        threshold = i * std_dictionary[stock1, stock2]
        pnl_threshold['PnLs'].append(
            pnl_dataframe[stock1 + str(threshold)].sum() + pnl_dataframe[stock2 + str(threshold)].sum())

pnl_threshold = pd.DataFrame(pnl_threshold)
pnl_threshold = pnl_threshold.set_index('Pairs')
pnl_threshold.to_excel('Thresholds.xlsx')

# Find Highest PnLs
highest_pnls = pnl_threshold.groupby(by='Pairs').agg({'PnLs' : max})
highest_pnls.sort_values('PnLs', ascending=False)

print(highest_pnls)


# Plot error-correction term (z-value) to observe what the spread looks like (see slide for comparison plot cointegrated pair)
def Plot_Thresholds(stock1, stock2):
    zvalue = tradable_pairs_data[stock1 + stock2, 'Z-Value']
    plt.figure(figsize=(20, 15))
    plt.xticks([])
    plt.title('Error-correction term stock pair ' + stock1 + ' and ' + stock2)
    zvalue.plot(alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    xmin = 0
    xmax = len(zvalue)

    # Boundries chosen to give an approximate good fit
    plt.hlines(pnl_threshold['Thresholds'][10:20], xmin, xmax, 'g')
    plt.hlines(-pnl_threshold['Thresholds'][10:20], xmin, xmax, 'r')

    plt.legend(['Z-Value', 'Positive Threshold', 'Negative Threshold'])
    plt.show()


Plot_Thresholds('ETH', 'BAB')


if analysis_type == 2:
    # Create a Plot that displays the Profitability of the Thresholds
    def profitability_of_the_thresholds(stock1, stock2):
        pnl_threshold[(pnl_threshold.index == (stock1, stock2))].plot(x='Thresholds', y='PnLs', figsize=(10,10))
        plt.title('Profitability of the Thresholds for ' + stock1 + ' and ' + stock2)
        plt.xlabel('Amount of Sigma away from the Mean')
        plt.ylabel('Profits and Losses')
        plt.legend(['Profits and Losses'])
        plt.grid()
        plt.show()


    profitability_of_the_thresholds('ETH', 'BAB')
    profitability_of_the_thresholds('BSV', 'ETH')
    profitability_of_the_thresholds('BSV', 'BTC')
    profitability_of_the_thresholds('BTC', 'LTC')

    # Determine the pairs and the threshold, manually chosen based on pnl_threshold and ensuring no overlap.
    threshold_dictionary = {('ETH', 'BAB'): 0.000183,
                            ('BSV', 'BTC'): 0.000075,
                            ('BSV', 'ETH'): 0.000050,
                            ('BTC', 'LTC'): 0.000394}

    print(threshold_dictionary)

    # Selection of the final pairs for this trading strategy
    stock_pairs_final = [('ETH', 'BAB'),
                         ('BSV', 'BTC'),
                         ('BSV', 'ETH'),
                         ('BTC', 'LTC')]

    print(stock_pairs_final)

    positions_strategy_1 = {}
    limit = 100

    for pair in stock_pairs_final:
        stock1 = pair[0]
        stock2 = pair[1]

        gamma = gamma_dictionary[stock1, stock2]

        threshold = threshold_dictionary[stock1, stock2]

        current_position_stock1 = 0
        current_position_stock2 = 0

        positions_strategy_1[stock1] = []

        for time, data_at_time in tradable_pairs_data.iterrows():

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

    # Set Ceiling (to prevent positions with not enough volume available) as well as define the timestamp
    positions_strategy_1 = np.ceil(pd.DataFrame(positions_strategy_1))
    positions_strategy_1['Timestamp'] = tradable_pairs_data.index
    positions_strategy_1 = positions_strategy_1.set_index('Timestamp')

    # The difference between the positions
    positions_diff_strategy_1 = positions_strategy_1.diff()[1:]

    # # Positions_diff first rows
    # positions_diff_strategy_1.head()

    #Used as mentioned earlier.
    positions_diff_strategy_1[-1:] = -positions_strategy_1[-1:]

    # Show Positions over Time
    for pairs in stock_pairs_final:
        stock1 = pairs[0]
        stock2 = pairs[1]

        plt.figure(figsize=(20, 5))

        positions_strategy_1[stock1].plot()
        positions_strategy_1[stock2].plot()

        plt.title('Positions over Time for ' + stock1 + ' and ' + stock2)
        plt.legend(["Position in " + stock1, "Position in " + stock2], loc='lower right')

        plt.show()

    pnl_dataframe_strategy_1 = pd.DataFrame()

    for pair in stock_pairs_final:
        stock1 = pair[0]
        stock2 = pair[1]

        Stock1_AskPrice = tradable_pairs_data[stock1, 'AskPrice'][1:]
        Stock1_BidPrice = tradable_pairs_data[stock1, 'BidPrice'][1:]
        Stock2_AskPrice = tradable_pairs_data[stock2, 'AskPrice'][1:]
        Stock2_BidPrice = tradable_pairs_data[stock2, 'BidPrice'][1:]

        pnl_dataframe_strategy_1[stock1] = np.where(positions_diff_strategy_1[stock1] > 0,
                                                    positions_diff_strategy_1[stock1] * -Stock1_BidPrice,
                                                    positions_diff_strategy_1[stock1] * -Stock1_AskPrice)
        pnl_dataframe_strategy_1[stock2] = np.where(positions_diff_strategy_1[stock2] > 0,
                                                    positions_diff_strategy_1[stock2] * -Stock2_BidPrice,
                                                    positions_diff_strategy_1[stock2] * -Stock2_AskPrice)

    print("The total profit is: €", round(pnl_dataframe_strategy_1.sum().sum()))

    pnl_dataframe_strategy_1['Timestamp'] = tradable_pairs_data.index[1:]
    pnl_dataframe_strategy_1 = pnl_dataframe_strategy_1.set_index('Timestamp')

    pnl_dataframe_strategy_1['PnL'] = pnl_dataframe_strategy_1.sum(axis=1)
    pnl_dataframe_strategy_1['Cum PnL'] = pnl_dataframe_strategy_1['PnL'].cumsum()

    for pair in stock_pairs_final:
        stock1 = pair[0]
        stock2 = pair[1]

        pnl_dataframe_strategy_1[stock1+stock2 + ' PnL'] = pnl_dataframe_strategy_1[stock1] + pnl_dataframe_strategy_1[stock2]
        pnl_dataframe_strategy_1[stock1+stock2 + ' Cum PnL'] = pnl_dataframe_strategy_1[stock1+stock2 + ' PnL'].cumsum()

    print(pnl_dataframe_strategy_1.tail())

    # All Pairs's PnL

    for pair in stock_pairs_final:
        stock1 = pair[0]
        stock2 = pair[1]

        pnl_dataframe_strategy_1[stock1 + stock2 + ' Cum PnL'].plot(figsize=(10, 10))
        plt.title('Cumulative PnL of ' + stock1 + ' and ' + stock2)
        plt.ylabel('Profit and Loss')
        plt.xlabel("")
        plt.grid()
        plt.xticks(rotation=20)
        plt.show()

    # All Pairs's PnLs (including total) in one graph

    pnl_dataframe_strategy_1['Cum PnL'].plot()

    for pair in stock_pairs_final:
        stock1 = pair[0]
        stock2 = pair[1]

        pnl_dataframe_strategy_1[stock1 + stock2 + ' Cum PnL'].plot(figsize=(10, 10))
        plt.legend(['Cum PnL', 'BB and JJ Cum PnL', 'FF and MM Cum PnL', 'DD and HH Cum PnL', 'AA and II Cum PnL'])
        plt.title('Cumulative PnLs of the Trading Strategy')
        plt.ylabel('Profit and Loss')
        plt.xlabel("")
        plt.grid()
        plt.xticks(rotation=20)
        plt.show()

    # Send to CSV
    pnl_dataframe_strategy_1.to_csv('Algorithm PnL Results.csv')

    # Send to Excel (in case CSV is incorrect)
    # pnl_dataframe_strategy_1.to_excel('Algorithm PnL Results.xlsx')
