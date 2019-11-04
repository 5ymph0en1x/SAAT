# SAAT - Statistical Arbitrage - Algorithmic Trading

## Guidance
* Create a new virtual environment (Python 3.6+) and install dependencies with 'pip install -r requirements.txt'.
* First, you need to acquire data from Bitfinex. This can be done by running 'caching_finex.py' (don't forget to input you credentials in /CONFIG/config_finex.yml).
* Second, run partial analysis of the data with 'analysis.py'. Once this is done, you'll get a file named 'Thresholds.xlsx'. Open it and look for the best thresholds for the best pairs.
* Input the settings found at the last step in 'analysis.py' at lines 494-503.
* Change parameter at line 9 to 2 and rerun 'analysis.py'. This will allow you to select the best pair possible by looking at the graphs generated.
* Input your choice in 'action.py' with the required parameters (lines 10-15).
* Run 'action.py' to start paper-trading with real-time data.

## Disclaimer
The article and the relevant codes and content are purely informative and none of the information provided constitutes any recommendation regarding any security, transaction or investment strategy for any specific person. The implementation described in the article could be risky and the market condition could be volatile and differ from the period covered above. All trading strategies and tools are implemented at the users’ own risk.
