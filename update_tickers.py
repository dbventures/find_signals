
# Commented out IPython magic to ensure Python compatibility.
import pandas_ta as pta
import yfinance as yf
# from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
# from pypfopt import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns
# from pypfopt import plotting
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
import numpy as np
import math
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt

import os
import requests

import pickle

# For parsing financial statements data from financialmodelingprep api
from urllib.request import urlopen
import json
def get_jsonparsed_data(url):
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)


# To reference specific dates when scraping for stock prices
import datetime
from dateutil.relativedelta import relativedelta

today = datetime.datetime.today()
string_today = today.strftime('%Y-%m-%d')
string_1y_ago = (today - relativedelta(years=1)).strftime('%Y-%m-%d')
string_4y_ago = (today - relativedelta(years=4)).strftime('%Y-%m-%d')
string_5d_ago = (today - relativedelta(days=5)).strftime('%Y-%m-%d')
# end_date = '2022-4-17'

# tickers = ['MA', 'META', 'V', 'AMZN', 'JPM', 'BA']
# #stocks_df = DataReader(tickers, 'yahoo', start = start_date, end = end_date)['Adj Close']
# #stocks_df = yf.download(tickers, start=start_date)

# df = yf.download("BA", start=start_date)
# df['Date'] = df.index
# df.head()


# get stock prices using yfinance library
def get_stock_price(symbol, freq = 'day'):
  if freq == 'week':
    df = yf.download(symbol, start=string_4y_ago, threads= False, progress=False, interval='1wk')
  elif freq == 'day':
    df = yf.download(symbol, start=string_1y_ago, threads= False, progress=False, interval='1d')
  elif freq == 'min':
    df = yf.download(symbol, start=string_5d_ago, threads= False, progress=False, interval='30m')
  df['Date'] = pd.to_datetime(df.index)
  df['Date'] = df['Date'].apply(mpl_dates.date2num)
  df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
  df['ATR20'] = pta.atr(df['High'], df['Low'], df['Close'], window=20, fillna=False, mamode = 'ema')
  df['SMA20'] = df.ta.sma(20)
  df['SMA50'] = df.ta.sma(50)
  df['SMA100'] = df.ta.sma(100)
  df['Ave Volume 20'] = df['Volume'].rolling(20).mean()
  return df

# get the full stock list of S&P 500
payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
stock_list_snp = payload[0]['Symbol'].values.tolist()
len(stock_list_snp)

# crypto_list = []

# url = 'https://web-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'

# params = {
#         'limit': 2000,
#     }

# r = requests.get(url, params=params)

# data = r.json()

# for number, item in enumerate(data['data']):
#     #print(f"{start+number:4} | {item['symbol']:5} | {item['date_added'][:10]}")
#     #print(item['symbol'])
#     crypto_list.append(item['symbol'] + '-USD')

# for symbol in crypto_list:
#     try:
#         get_stock_price(symbol)
#     except:
#         crypto_list.remove(symbol)

# print(len(crypto_list))

# with open("crypto_list.txt", 'w') as f:
#     for ticker in crypto_list:
#         f.write(str(ticker) + '\n')

# crypto_list = []

# url = 'https://web-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'

# params = {
#         'limit': 80,
#     }

# r = requests.get(url, params=params)

# data = r.json()

# for number, item in enumerate(data['data']):
#     #print(f"{start+number:4} | {item['symbol']:5} | {item['date_added'][:10]}")
#     #print(item['symbol'])
#     if "US" not in item['symbol']:
#         crypto_list.append(item['symbol'] + '-USD')

# for symbol in crypto_list:
#     try:
#         get_stock_price(symbol)
#     except:
#         crypto_list.remove(symbol)

# print(len(crypto_list))

# # with open("crypto_list.txt", 'w') as f:
# #     for ticker in crypto_list:
# #         f.write(str(ticker) + '\n')

futures_list = ['YM=F','ES=F','NQ=F','TF=F', # index
                'CL=F','NG=F', # energy
                'GC=F','SI=F','HG=F','PL=F','PA=F', # metal
                '6E=F','6B=F','6A=F','6C=F','6S=F','6J=F'] # forex

forex_list = ['EURUSD=T','GBPUSD=T','AUDUSD=T','USDJPY=T','CHFUSD=T','CADUSD=T','NZDUSD=T',
              'EURGBP=T','EURJPY=T','EURAUD=T','GBPJPY=T','GBPAUD=T','AUDNZD=T','AUDJPY=T']

# os.environ['FMP_API_KEY'] = 'b187d0baf2c855400870f859dac36b99'
# apiKey = os.environ['FMP_API_KEY']
# url = f"https://financialmodelingprep.com/api/v3/available-traded/list?apikey={apiKey}"
# df_all_stocks = pd.DataFrame(get_jsonparsed_data(url))
# df_all_stocks.to_csv("all_stocks_info.csv")
# df_all_stocks

# df_all_stocks = pd.read_csv("all_stocks_info.csv")
# #print(df_all_stocks.exchange.value_counts()[:20]) # print top exchanges by stocks
# stock_exchanges = ["NASDAQ", "New York Stock Exchange", "HKSE"]
# df_stocks_filtered = df_all_stocks[(df_all_stocks["exchange"].isin(stock_exchanges)) & (df_all_stocks["type"]=='stock')] #9k+ stocks here already
# stock_list = list(df_stocks_filtered.symbol.unique())
# print(len(stock_list))

# with open("stock_list.txt", 'w') as f:
#     for ticker in stock_list:
#         f.write(str(ticker) + '\n')

# crypto_list = []
# with open("crypto_list.txt", 'r') as f:
#     for line in f:
#         crypto_list.append(line.rstrip('\n'))

stock_list = []
with open("stock_list.txt", 'r') as f:
    for line in f:
        stock_list.append(line.rstrip('\n'))

len(stock_list)

stock_list_all = set(stock_list_snp).union(set(stock_list))


#method 1: fractal candlestick pattern
# determine bullish fractal
def is_support(df,i):
    cond1 = df['Low'][i] < df['Low'][i-1]
    cond2 = df['Low'][i] < df['Low'][i+1]
    cond3 = df['Low'][i+1] < df['Low'][i+2]
    cond4 = df['Low'][i-1] < df['Low'][i-2]

    cond_3bar_1 = (df['Low'][i-1] - df['Low'][i]) > 0.8*df['ATR20'][i]
    cond_3bar_2 = (df['Low'][i+1] - df['Low'][i]) > 0.8*df['ATR20'][i]


    return (cond1 and cond2 and cond3 and cond4) or (cond1 and cond2 and (cond_3bar_1 or cond_3bar_2))

# determine bearish fractal
def is_resistance(df,i):
    cond1 = df['High'][i] > df['High'][i-1]
    cond2 = df['High'][i] > df['High'][i+1]
    cond3 = df['High'][i+1] > df['High'][i+2]
    cond4 = df['High'][i-1] > df['High'][i-2]

    cond_3bar_1 = (df['High'][i] - df['High'][i-1]) > 0.6*df['ATR20'][i]
    cond_3bar_2 = (df['High'][i] - df['High'][i+1]) > 0.6*df['ATR20'][i]

    return (cond1 and cond2 and cond3 and cond4) or (cond1 and cond2 and (cond_3bar_1 or cond_3bar_2))




# to make sure the new level area does not exist already
def is_far_from_level(value, levels, df):
  ave =  np.mean(df['High'] - df['Low'])
  return np.sum([abs(value-level) < ave for _,level in levels]) == 0

def remove_previous(value, levels, low_or_high):
  to_remove = []
  for level in levels:
    if low_or_high == 'low': # remove previous swing low if it already broke previous
      if value <= level[1]: # level is a tuple
        to_remove.append(level) # cannot remove directly from levels as we are still looping through it
    elif low_or_high == 'high': # remove previous swing high if it already broke previous
      if value >= level[1]:
        to_remove.append(level)
  for remove in to_remove:
    levels.remove(remove)

def find_levels(df):
    # a list to store resistance and support levels
    levels_low = []
    levels_high = []
    #for i in range(2, df.shape[0] - 2): # exclude most recent 10 bars, anyway the -2 is cos need 2 more bars in the fractal

    for i in range(2, df.shape[0] - 12): # exclude most recent 10 bars

        if is_support(df, i):
            low = df['Low'][i]
            #if is_far_from_level(low, levels_low, df):
            # do not use most recent swing high or low to remove it, because the most recent one may be a force top/bottom but it overrides the previous one
            # either that or just don't find LP for most recent 10 bars in the range above.
            remove_previous(low, levels_low, 'low') # delete previous if this swing low is below previous, as previous is broken
            levels_low.append((df.index[i], low))

        elif is_resistance(df, i):
            high = df['High'][i]
            #if is_far_from_level(high, levels_high, df):
            remove_previous(high, levels_high, 'high')
            levels_high.append((df.index[i], high))

    return levels_low, levels_high

# v5, doesnt make sense to have more than swing 4, because swing 5 onwards is too late if swing 5 is below LP, since must be within 4 bar close back above
# to check: lowest(low, 2) in TOS means today and yesterday only, 2 is including the current bar
# lowest(low[3], 3) means from 3 days ago to 5 days ago, 5, 4, 3, days including 3
# define the minimum low for the last length bars including the current bar
# return drop range to plot?
# it makes sense to test for force bottom/top for last few bars in separate function only after this test passes
# only find lp levels any of the ur/dr/fs passes
# characteristic: sideways price action movement
# entry rules 1. final price movement as majority flush down to force lp (bottom)
# entry rules 2. bullish exe to close at or above LP within 4 bars (so if swing 5, then final bar at or above) (means any of the 5 bars below lp force bottom)
# exit rule, SL below exe, or recent swing low, TP to project, to enter based on exe, half of exe from high to mid of exe

# TODO: good to include at LP it touches to current price, what is the lowest, is it 50%
# TODO: 1 bar can also be UR, is this already in the code?
def bullish_dr1(df,i, drop_days=3): # i should be -1 for most recent setup, for backtesting, can be changed
  exe = (df['Close'][i] - df['Low'][i])/(df['High'][i] - df['Low'][i]) >= 2/3 # this is a boolean, bullish pin and bullish ice cream both covered
  price_cond = df['Close'][i] > 2
  vol = (df['Ave Volume 20'][i] > 100000) and (df['Volume'][i] > 100000)
  #drop_days = 3 # smaller means steeper
  j = drop_days - 3 #45678 is already 3 days difference so adjust it
  if i == -1:
    swing_1 = (df['Low'][i] < df['Low'][i-3:i].min()) and ((df['High'][i-3-j:].max() - df['Low'][i-3-j:].min()) >= 4*df['ATR20'][i]) # this is also bar that comes back up as exe
    # 1 day ago bar is lowest, lower than today bar low, also lower than past 4 bars (5 days ago to 2 days ago), and 1 day ago compared to 6 day ago range is higher than 4 ATR20
    # maybe don't just use 3 days ago for second part, use 6 days instead, or like 5 days at least, i.e. 678910 instead of 45678
    # second part should be same as third part basically, 45678 means 3 days before
    swing_2 = (df['Low'][i-1] < df['Low'][i])          and (df['Low'][i-1] < df['Low'][i-4:i-1].min()) and ((df['High'][i-4-j:i].max() - df['Low'][i-4-j:i].min()) >= 4*df['ATR20'][i])
    swing_3 = (df['Low'][i-2] < df['Low'][i-1:].min()) and (df['Low'][i-2] < df['Low'][i-5:i-2].min()) and ((df['High'][i-5-j:i-1].max() - df['Low'][i-5-j:i-1].min()) >= 4*df['ATR20'][i])
    swing_4 = (df['Low'][i-3] < df['Low'][i-2:].min()) and (df['Low'][i-3] < df['Low'][i-6:i-3].min()) and ((df['High'][i-6-j:i-2].max() - df['Low'][i-6-j:i-2].min()) >= 4*df['ATR20'][i])
    # swing_5 = (df['Low'][i-4] < df['Low'][i-3:].min()) and (df['Low'][i-4] < df['Low'][i-7:i-4].min()) and ((df['High'][i-7-j:i-3].max() - df['Low'][i-7-j:i-3].min()) >= 4*df['ATR20'][i])
    # swing_6 = (df['Low'][i-5] < df['Low'][i-4:].min()) and (df['Low'][i-5] < df['Low'][i-8:i-5].min()) and ((df['High'][i-8-j:i-4].max() - df['Low'][i-8-j:i-4].min()) >= 4*df['ATR20'][i])
  else:
    swing_1 = (df['Low'][i] < df['Low'][i-3:i-1+1].min()) and ((df['High'][i-5:i+1].max() - df['Low'][i-5:i+1].min()) >= 4*df['ATR20'][i])
    swing_2 = (df['Low'][i-1] < df['Low'][i])             and (df['Low'][i-1] < df['Low'][i-4:i-1].min()) and ((df['High'][i-4-j:i-1].max() - df['Low'][i-4-j:i-1].min()) >= 4*df['ATR20'][i])
    swing_3 = (df['Low'][i-2] < df['Low'][i-1:i+1].min()) and (df['Low'][i-2] < df['Low'][i-5:i-2].min()) and ((df['High'][i-5-j:i-2].max() - df['Low'][i-5-j:i-2].min()) >= 4*df['ATR20'][i])
    swing_4 = (df['Low'][i-3] < df['Low'][i-2:i+1].min()) and (df['Low'][i-3] < df['Low'][i-6:i-3].min()) and ((df['High'][i-6-j:i-3].max() - df['Low'][i-6-j:i-3].min()) >= 4*df['ATR20'][i])
    # swing_5 = (df['Low'][i-4] < df['Low'][i-3:i+1].min()) and (df['Low'][i-4] < df['Low'][i-7:i-4].min()) and ((df['High'][i-7-j:i-4].max() - df['Low'][i-7-j:i-4].min()) >= 4*df['ATR20'][i])
    # swing_6 = (df['Low'][i-5] < df['Low'][i-4:i+1].min()) and (df['Low'][i-5] < df['Low'][i-8:i-5].min()) and ((df['High'][i-8-j:i-5].max() - df['Low'][i-8-j:i-5].min()) >= 4*df['ATR20'][i])

  which_swing = None
  if swing_1:
    which_swing = 1
  if swing_2:
    which_swing = 2
  if swing_3:
    which_swing = 3
  if swing_4:
    which_swing = 4
  # if swing_5:
  #   which_swing = 5
  # if swing_6:
  #   which_swing = 6
  # use which swing to decide which bars onwards to look for lp force top/bottom

  return ((exe and price_cond and vol and (swing_1 or swing_2 or swing_3 or swing_4
                                  #or swing_5 or swing_6
                                  )),
        which_swing)

# v5
# characteristic: sideways price action movement
# entry rules 1. final price movement as majority flush up to force lp (top)
# entry rules 2. bearish exe to CLOSE at or below LP within 4 bars (so if swing 5, then final bar at or above) (means any of the 5 bars above lp force top)
# exit rule, SL above exe, or recent swing high, TP to project, to enter based on exe, half of exe from high to mid of exe

def bearish_ur1(df,i,rise_days=3): # i should be -1 for most recent setup, for backtesting, can be changed
  exe = (df['Close'][i] - df['Low'][i])/(df['High'][i] - df['Low'][i]) <= 1/3 # this is a boolean, bearish pin
  price_cond = df['Close'][i] > 2
  vol = (df['Ave Volume 20'][i] > 100000) and (df['Volume'][i] > 100000)
  #rise_days = 3 # smaller means steeper
  j = rise_days - 3 #45678 is already 3 days difference so adjust it

  if i == -1:
    swing_1 = (df['High'][i] > df['High'][i-3:i].max()) and ((df['High'][i-3-j:].max() - df['Low'][i-3-j:].min()) >= 4*df['ATR20'][i])
    # see bullish_dr same part of function for notes
    swing_2 = (df['High'][i-1] > df['High'][i])          and (df['High'][i-1] > df['High'][i-4:i-1].max()) and ((df['High'][i-4-j:i-1].max() - df['Low'][i-4-j:i-1].min()) >= 4*df['ATR20'][i])
    swing_3 = (df['High'][i-2] > df['High'][i-1:].max()) and (df['High'][i-2] > df['High'][i-5:i-2].max()) and ((df['High'][i-5-j:i-2].max() - df['Low'][i-5-j:i-2].min()) >= 4*df['ATR20'][i])
    swing_4 = (df['High'][i-3] > df['High'][i-2:].max()) and (df['High'][i-3] > df['High'][i-6:i-3].max()) and ((df['High'][i-6-j:i-3].max() - df['Low'][i-6-j:i-3].min()) > 4*df['ATR20'][i])
    #swing_5 = (df['High'][i-4] > df['High'][i-3:].max()) and (df['High'][i-4] > df['High'][i-7:i-4].max()) and ((df['High'][i-7-j:i-4].max() - df['Low'][i-7-j:i-4].min()) >= 4*df['ATR20'][i])
    #swing_6 = (df['High'][i-5] > df['High'][i-4:].max()) and (df['High'][i-5] > df['High'][i-8:i-5].max()) and ((df['High'][i-8-j:i-5].max() - df['Low'][i-8-j:i-5].min()) >= 4*df['ATR20'][i])
  else:
    swing_1 = (df['High'][i] > df['High'][i-4:i-1+1].max()) and ((df['High'][i-6:i+1].max() - df['Low'][i-6:i+1].min()) >= 4*df['ATR20'][i])
    swing_2 = (df['High'][i-1] > df['High'][i])             and (df['High'][i-1] > df['High'][i-4:i-1].max()) and ((df['High'][i-4-j:i-1].max() - df['Low'][i-4-j:i-1].min()) >= 4*df['ATR20'][i])
    swing_3 = (df['High'][i-2] > df['High'][i-1:i+1].max()) and (df['High'][i-2] > df['High'][i-5:i-2].max()) and ((df['High'][i-5-j:i-2].max() - df['Low'][i-5-j:i-2].min()) >= 4*df['ATR20'][i])
    swing_4 = (df['High'][i-3] > df['High'][i-2:i+1].max()) and (df['High'][i-3] > df['High'][i-6:i-3].max()) and ((df['High'][i-6-j:i-3].max() - df['Low'][i-6-j:i-3].min()) > 4*df['ATR20'][i])
    #swing_5 = (df['High'][i-4] > df['High'][i-3:i+1].max()) and (df['High'][i-4] > df['High'][i-7:i-4].max()) and ((df['High'][i-7-j:i-4].max() - df['Low'][i-7-j:i-4].min()) >= 4*df['ATR20'][i])
    #swing_6 = (df['High'][i-5] > df['High'][i-4:i+1].max()) and (df['High'][i-5] > df['High'][i-8:i-5].max()) and ((df['High'][i-8-j:i-5].max() - df['Low'][i-8-j:i-5].min()) >= 4*df['ATR20'][i])

  which_swing = None
  if swing_1:
    which_swing = 1
  if swing_2:
    which_swing = 2
  if swing_3:
    which_swing = 3
  if swing_4:
    which_swing = 4
#   if swing_5:
#     which_swing = 5
#   if swing_6:
#     which_swing = 6
  #use which swing to decide which bars onwards to look for lp force top/bottom

  return ((exe and price_cond and vol and (swing_1 or swing_2 or swing_3 or swing_4
                                  #or swing_5 or swing_6
                                  )),
        which_swing)

def bearish_fs(df,i): # combine with force top or test SMA20/50
  # maybe out rectangle to draw
  exe = (df['Close'][i] - df['Low'][i])/(df['High'][i] - df['Low'][i]) <= 1/3 # this is a boolean, bearish pin
  price_cond = df['Close'][i] > 2
  vol = (df['Ave Volume 20'][i] > 100000) and (df['Volume'][i] > 100000)

  fs_3_bar = (((df['Low'][i-2] <= df['Low'][i-1]) and (df['Low'][i-2] <= df['Low'][i]))
            and (df['High'][i-2] >= df['High'][i-1]) and (df['High'][i] > df['High'][i-2])
            and (df['Close'][i] <= df['High'][i-2]))

  fs_4_bar = (((df['Low'][i-3] <= df['Low'][i-2]) and (df['Low'][i-3] <= df['Low'][i-1]) and (df['Low'][i-3] <= df['Low'][i]))
                and (df['High'][i-3] >= df['High'][i-2]) and ((df['High'][i] > df['High'][i-3]) or (df['High'][i-1] > df['High'][i-3]))
                and (df['Close'][i] <= df['High'][i-3]))
  fs_5_bar = (((df['Low'][i-4] <= df['Low'][i-3]) and (df['Low'][i-4] <= df['Low'][i-2]) and (df['Low'][i-4] <= df['Low'][i-1]) and (df['Low'][i-4] <= df['Low'][i]))
                and (df['High'][i-4] >= df['High'][i-3]) and ((df['High'][i] > df['High'][i-4]) or (df['High'][i-1] > df['High'][i-4]) or (df['High'][i-2] > df['High'][i-4]))
                and (df['Close'][i] <= df['High'][i-4]))
  fs_6_bar = (((df['Low'][i-5] <= df['Low'][i-4]) and (df['Low'][i-5] <= df['Low'][i-3]) and (df['Low'][i-5] <= df['Low'][i-2]) and (df['Low'][i-5] <= df['Low'][i-1]) and (df['Low'][i-5] <= df['Low'][i]))
                and (df['High'][i-5] >= df['High'][i-4]) and ((df['High'][i] > df['High'][i-5]) or (df['High'][i-1] > df['High'][i-5]) or (df['High'][i-2] > df['High'][i-5]) or (df['High'][i-3] > df['High'][i-5]))
                and (df['Close'][i] <= df['High'][i-5]))

  which_bar= None
  if fs_3_bar:
    which_bar = 3
  if fs_4_bar:
    which_bar = 4
  if fs_5_bar:
    which_bar = 5
  if fs_6_bar:
    which_bar = 6

  return (exe and price_cond and vol and (fs_3_bar or fs_4_bar or fs_5_bar or fs_6_bar)), which_bar
  #return (test or fs_3_bar or fs_4_bar or fs_5_bar or fs_6_bar)

def bullish_fs(df,i): # combine with force bottom or test SMA20/50
  exe = (df['Close'][i] - df['Low'][i])/(df['High'][i] - df['Low'][i]) >= 2/3 # this is a boolean, bullish pin
  price_cond = df['Close'][i] > 2
  vol = (df['Ave Volume 20'][i] > 100000) and (df['Volume'][i] > 100000)

  fs_3_bar = ((df['High'][i-2] >= df['High'][i-1]) and (df['High'][i-2] >= df['High'][i])) and (df['Low'][i-2] <= df['Low'][i-1]) and (df['Low'][i] < df['Low'][i-2]) and (df['Close'][i] >= df['Low'][i-2])
  fs_4_bar = ((df['High'][i-3] >= df['High'][i-2]) and (df['High'][i-3] >= df['High'][i-1]) and (df['High'][i-3] >= df['High'][i])) and (df['Low'][i-3] <= df['Low'][i-2]) and ((df['Low'][i] < df['Low'][i-3]) or (df['Low'][i-1] < df['Low'][i-3])) and (df['Close'][i] >= df['Low'][i-3])
  fs_5_bar = ((df['High'][i-4] >= df['High'][i-3]) and (df['High'][i-4] >= df['High'][i-2]) and (df['High'][i-4] >= df['High'][i-1]) and (df['High'][i-4] >= df['High'][i])) and (df['Low'][i-4] <= df['Low'][i-3]) and ((df['Low'][i] < df['Low'][i-4]) or (df['Low'][i-1] < df['Low'][i-4]) or (df['Low'][i-2] < df['Low'][i-4])) and (df['Close'][i] >= df['Low'][i-4])
  fs_6_bar = ((df['High'][i-5] >= df['High'][i-4]) and (df['High'][i-5] >= df['High'][i-3]) and (df['High'][i-5] >= df['High'][i-2]) and (df['High'][i-5] >= df['High'][i-1]) and (df['High'][i-5] >= df['High'][i])) and (df['Low'][i-5] <= df['Low'][i-4]) and ((df['Low'][i] < df['Low'][i-5]) or (df['Low'][i-1] < df['Low'][i-5]) or (df['Low'][i-2] < df['Low'][i-5]) or (df['Low'][i-3] < df['Low'][i-5])) and (df['Close'][i] >= df['Low'][i-5])

  which_bar= None
  if fs_3_bar:
    which_bar = 3
  if fs_4_bar:
    which_bar = 4
  if fs_5_bar:
    which_bar = 5
  if fs_6_bar:
    which_bar = 6

  return (exe and price_cond and vol and (fs_3_bar or fs_4_bar or fs_5_bar or fs_6_bar)), which_bar # turn to dict better
  #return (test or fs_3_bar or fs_4_bar or fs_5_bar or fs_6_bar)

# for force bottom, minimum of all lows go below LP (support level), then final close is above then ok



def test_force_top(df, levels): # should give it levels_high, 5th bar should be below, 4th bar onwards anything above
  for level in levels:
      not_too_late = df['High'][-5] < level[1] # 5th bar should be still below, haven't break yet, if not, its too late
      went_above = df['High'][-4:].max() > level[1] # 4 bars at any point went above level
      close_below = df['Close'][-1] <= level[1] # last bar close back below or equal
      #print(f"Testing for levels {level}, 5th bar haven't break {not_too_late}, went above {went_above}, close below {close_below}")
      if not_too_late and went_above and close_below:
        return True, level # must know which is the level to plot it out
  return False, False

def test_force_bottom(df, levels): # should give it levels_low
  for level in levels:
      not_too_late = df['Low'][-5] > level [1]
      went_below = df['Low'][-4:].min() < level[1]
      close_above = df['Close'][-1] >= level[1]
      #print(f"Testing for levels {level}, 5th bar haven't break {not_too_late}, went below {went_below}, close above {close_above}")
      if not_too_late and went_below and close_above:
        return True, level # must know which is the level to plot it out
  return False, False

def check_50_steepness_top(df, swing_bar, level, rise_days = 3):
    start_of_swing = -swing_bar - 3 - rise_days
    return (level[1] - df['Low'][start_of_swing:-swing_bar].min()) > 0.5*(level[1] - (df.loc[level[0]:df.index[start_of_swing]]['Low'].min()))

def check_50_steepness_bottom(df, swing_bar, level, drop_days = 3):
    start_of_swing = -swing_bar - 3 - drop_days
    return ((df['High'][start_of_swing:-swing_bar].max() - level[1]) > 0.5*((df.loc[level[0]:df.index[start_of_swing]]['High'].max()) - level[1]))

# # add criteria, entry and stop loss less than 1 ATR?
# can write streamlit app to calculate price also
# add override enter price?
def get_enter_prices(df, ticker, direction = 'Long', risk = 300, currency = 'USD', ratio = 2, enter_override = None, stop_loss_buffer = 0.02): # should give it levels_low
    if ticker[-2:] == 'HK':
        currency = 'HKD'
    else:
        currency = 'USD'

    if currency == 'USD':
        risk*=0.75
    elif currency == 'HKD':
        risk*=5.82
    more_than_atr = False
    price_dict = {}
    if direction == 'Long':

        enter_dict = {
            '0.25':((df['High'][-1] - df['Low'][-1]) * 0.75) + df['Low'][-1],
            '0.5':((df['High'][-1] - df['Low'][-1]) * 0.5) + df['Low'][-1],
            'close': df['Close'][-1]
            }

        if enter_override:
            enter_dict['override'] = enter_override
        for key, enter in enter_dict.items():
            stop_loss = df['Low'][-5:].min() - stop_loss_buffer
            take_profit = (enter - stop_loss) * ratio + enter
            n_shares = risk/(enter - stop_loss)
            if (enter - stop_loss) > df['ATR20'][-1]:
                more_than_atr = True
            price_dict[key] = {'enter': enter,
                  'take_profit': take_profit,
                  'stop_loss': stop_loss,
                  'n_shares': n_shares,
                  'more_than_atr': more_than_atr}

    elif direction == 'Short':
        enter_dict = {
            '0.25':((df['High'][-1] - df['Low'][-1]) * 0.25) + df['Low'][-1],
            '0.5':((df['High'][-1] - df['Low'][-1]) * 0.5) + df['Low'][-1],
            'close': df['Close'][-1]
            }

        if enter_override:
            enter_dict['override'] = enter_override
        for key, enter in enter_dict.items():
            stop_loss = df['High'][-5:].max() + stop_loss_buffer
            take_profit = enter - (stop_loss - enter) * ratio
            n_shares = risk/(stop_loss - enter)
            if (stop_loss - enter) > df['ATR20'][-1]:
                more_than_atr = True

            price_dict[key] = {'enter': enter,
                          'take_profit': take_profit,
                          'stop_loss': stop_loss,
                          'n_shares': n_shares,
                          'more_than_atr': more_than_atr}


    return price_dict

# test_sma_above
def test_sma_above(df, i, j): # combine with force bottom or trend
  exe = (df['Close'][i] - df['Low'][i])/(df['High'][i] - df['Low'][i]) >= 2/3 # this is a boolean, bullish pin
  price_cond = df['Close'][i] > 2
  trend_cond = df['SMA20'][i] > df['SMA50'][i] # current price can be below SMA, need not be above, just need to be near
  trend_cond2 = df['SMA50'][i] > df['SMA100'][i]
  bars_buffer = 10/100 * df['ATR20'][i] # exponential ATR?
  exe_buffer = 1 * df['ATR20'][i]

  # final bar higher than SMA, or at least very near to SMA (10/100 of ATR as shown above)
  exe_with_flow = (df['High'][i] > df['SMA50'][i]) or (abs(df['High'][i] - df['SMA50'][i]) < bars_buffer)
  exe_not_too_far_from_ma = (abs(df['High'][i] - df['SMA20'][i]) < exe_buffer) or (abs(df['High'][i] - df['SMA50'][i]) < exe_buffer)

  # any bar from current i to j bars ago is near, will match
  near_ma = False
  for k in range(i, j-1, -1):
      near_ma = (((df['High'][k] > df['SMA20'][k]) and (df['Low'][k] < df['SMA20'][k])) # bar cuts 20 SMA
      or (abs(df['High'][k] - df['SMA20'][k]) < bars_buffer) # high not too far from 20 SMA
      or (abs(df['Low'][k] - df['SMA20'][k]) < bars_buffer) # low not too far from 20 SMA
      or ((df['High'][k] > df['SMA50'][k]) and (df['Low'][k] < df['SMA50'][k]))
      or (abs(df['High'][k] - df['SMA50'][k]) < bars_buffer)
      or (abs(df['Low'][k] - df['SMA50'][k]) < bars_buffer))
      if near_ma:
            break

  return (exe and price_cond and trend_cond and trend_cond2 and exe_with_flow and exe_not_too_far_from_ma and near_ma)



# test_sma_below
def test_sma_below(df, i, j): # combine with force bottom or trend
  exe = (df['Close'][i] - df['Low'][i])/(df['High'][i] - df['Low'][i]) <= 1/3 # this is a boolean, bearish pin
  price_cond = df['Close'][i] > 2
  # maybe add 50 < 100 also
  trend_cond = df['SMA20'][i] < df['SMA50'][i] # current price can be below SMA, need not be above, just need to be near
  trend_cond2 = df['SMA50'][i] < df['SMA100'][i]
  bars_buffer = 10/100 * df['ATR20'][i] # exponential ATR?
  exe_buffer = 1 * df['ATR20'][i]

  # final bar low lower than SMA, or at least very near to SMA (10/100 of ATR as shown above)
  exe_with_flow = (df['Low'][i] < df['SMA50'][i]) or (abs(df['Low'][i] - df['SMA50'][i]) < bars_buffer)
  exe_not_too_far_from_ma = (abs(df['Low'][i] - df['SMA20'][i]) < exe_buffer) or (abs(df['Low'][i] - df['SMA50'][i]) < exe_buffer)

  # any bar from current i to j bars ago is near, will match
  near_ma = False
  for k in range(i, j-1, -1): # j-1 as last value not included
      near_ma = (((df['High'][k] > df['SMA20'][k]) and (df['Low'][k] < df['SMA20'][k])) # bar cuts 20 SMA
      or (abs(df['High'][k] - df['SMA20'][k]) < bars_buffer) # high not too far from 20 SMA
      or (abs(df['Low'][k] - df['SMA20'][k]) < bars_buffer) # low not too far from 20 SMA
      or ((df['High'][k] > df['SMA50'][k]) and (df['Low'][k] < df['SMA50'][k]))
      or (abs(df['High'][k] - df['SMA50'][k]) < bars_buffer)
      or (abs(df['Low'][k] - df['SMA50'][k]) < bars_buffer))
      if near_ma:
            break

  return (exe and price_cond and trend_cond and trend_cond2 and exe_with_flow and exe_not_too_far_from_ma and near_ma)

# here onwards need SMA

def find_recent_levels(df, i, j):
    # a list to store resistance and support levels
    recent_lows = []
    recent_highs = []
    #for i in range(2, df.shape[0] - 2): # exclude most recent 10 bars, anyway the -2 is cos need 2 more bars in the fractal

    # from most recent to some bars earlier, maybe i can be -10, j can be -30 for e.g., i cannot be -1, need 2 bars after at least
    for k in range(i, j-1, -1):

        if is_support(df, k):
            low = df['Low'][k]
            #if is_far_from_level(low, levels_low, df):
            # do not use most recent swing high or low to remove it, because the most recent one may be a force top/bottom but it overrides the previous one
            # either that or just don't find LP for most recent 10 bars in the range above.
            remove_previous(low, levels_low, 'low') # delete previous if this swing low is below previous, as previous is broken
            recent_lows.append((k, low))

        elif is_resistance(df, k):
            high = df['High'][k]
            #if is_far_from_level(high, levels_high, df):
            remove_previous(high, levels_high, 'high')
            recent_highs.append((k, high))


    return recent_lows, recent_highs

# any bar between sma start and end test sma
# recent swing has start and end too
def bullish_uc1(df, i, sma_start = -1, sma_end = -6, recent_swing_start = -6, recent_swing_end = -26):
    above_sma = test_sma_above(df, sma_start, sma_end)
    if above_sma: # don't bother testing for lows and highs if not even testing sma
        recent_lows, recent_highs = find_recent_levels(df, recent_swing_start, recent_swing_end)
        if (len(recent_lows) > 0) and (len(recent_highs) > 0):
            most_recent_low = recent_lows[0]
            most_recent_high = recent_highs[0]
            most_recent_low_index = most_recent_low[0] # first one that went into the list
            most_recent_high_index = most_recent_high[0]
            # we want this, low occured first then high, bullish, so can check for force bottom
            if most_recent_low_index < most_recent_high_index:

                # exe already tested in above_sma
                if df['Close'][i] >= most_recent_low[1]: # close above recent low, the low of candle can still be below
                    # print("stage 2") # alot managed to clear this stage
                    # before that must go below recent low first
                    # delay means number of candles that were below the low before it closed back up
                    # exe is also the candle that went below and back up, one day before is completely above swing low and is lowest of all candles from swing high onwards
                    #no_delay = (df['Low'][i] < most_recent_low[1]) and (df['Low'][i-1] > most_recent_low[1]) and (df['Low'][i-1] < df['Low'][most_recent_high:i].min())  most_recent_low[1])
                    #got_1_delay = (df['Low'][i-1] < most_recent_low[1]) and (df['Low'][i-2] > most_recent_low[1]) and (df['Low'][i-2] < df['Low'][most_recent_high:i-1].min())  most_recent_low[1])

                    delay_cond = False

                    for d in range(0, 5):
                        #print("stage 3")
                        # actually must test if any flush up bar 3 bars leading up to this bar
                        if most_recent_high_index < i-d: # most recent high must be at a date before the delay candles
                            # maybe just df['Low'][most_recent_high:i-d-1].min() > most_recent_low[1]
                            #delay_cond = (df['Low'][i-d] < most_recent_low[1]) and (df['Low'][i-d-1] > most_recent_low[1]) and (df['Low'][i-d-1] < df['Low'][most_recent_high:i-d-1].min())
                            delay_cond = (df['Low'][i-d] < most_recent_low[1]) and (df['Low'][most_recent_high_index:i-d].min() > most_recent_low[1])
                            #delay_cond = (df['Low'][i-d] < most_recent_low[1])
                            if delay_cond:
                                #print("Bullish UC1 stage")
                                print(delay_cond, (df.index[most_recent_low_index], most_recent_low[1]), (df.index[most_recent_high_index], most_recent_high[1]))
                                return delay_cond, (df.index[most_recent_low_index], most_recent_low[1]), (df.index[most_recent_high_index], most_recent_high[1])
    return False, None, None # anything not met, return False


# any bar between sma start and end test sma
# recent swing has start and end too
def bearish_dc1(df, i, sma_start = -1, sma_end = -6, recent_swing_start = -6, recent_swing_end = -26):
    below_sma = test_sma_below(df, sma_start, sma_end)
    if below_sma: # don't bother testing for lows and highs if not even testing sma
        recent_lows, recent_highs = find_recent_levels(df, recent_swing_start, recent_swing_end)
        if (len(recent_lows) > 0) and (len(recent_highs) > 0):
            most_recent_low = recent_lows[0]
            most_recent_high = recent_highs[0]
            most_recent_low_index = most_recent_low[0] # first one that went into the list
            most_recent_high_index = most_recent_high[0]
            # we want this, high occured first then low, bearish, then now high again, so can check for force top
            if most_recent_high_index < most_recent_low_index:

                # exe already tested in below_sma
                if df['Close'][i] <= most_recent_high[1]: # close back below recent high, the high of candle can still be above
                    # before that must go above recent high first
                    # delay means number of candles that were above the high before it closed back down
                    # exe is also the candle that went up and back down, one day before is completely below swing high and is highest of all candles from swing low onwards
                    #no_delay = (df['Low'][i] < most_recent_low[1]) and (df['Low'][i-1] > most_recent_low[1]) and (df['Low'][i-1] < df['Low'][most_recent_high:i].min())  most_recent_low[1])
                    #got_1_delay = (df['Low'][i-1] < most_recent_low[1]) and (df['Low'][i-2] > most_recent_low[1]) and (df['Low'][i-2] < df['Low'][most_recent_high:i-1].min())  most_recent_low[1])

                    delay_cond = False

                    for d in range(0, 5):
                        #print("stage 3")

                        if most_recent_low_index < i-d: # most recent low must be at a date before the delay candles
                            # maybe just df['Low'][most_recent_high:i-d-1].min() > most_recent_low[1]
                            # delay_cond = (df['Low'][i-d] < most_recent_low[1]) and (df['Low'][i-d-1] > most_recent_low[1]) and (df['Low'][i-d-1] < df['Low'][most_recent_high_index:i-d-1].min())
                            delay_cond = (df['High'][i-d] > most_recent_high[1]) and (df['High'][most_recent_high_index:i-d].max() < most_recent_high[1])
                            if delay_cond:
                                return delay_cond, (df.index[most_recent_low_index], most_recent_low[1]), (df.index[most_recent_high_index], most_recent_high[1])
    return False, None, None # anything not met, return False

len(stock_list_all)

# TODO: make script for crypt as additional page
# make script for short term only on S&P DR1X # done
# make function for test 20/50MA from above
# make function for test 20/50MA from below

# bearish/bullish FS reversal: Done
# UR1/DR1: Done
# UR2/DR2 (don't need to program more?) # check for MA and return those

# bearish/bullish FS test MA
# both below similar
# UC1/DC1
# UC2/DC2

# script to scan everything for futures with position sizing
# script to scan everything for forex 4h timeframe with position sizing

day = -1 # minus 1 means most recent date
freq = 'day'
rise_drop_days = 5
sma_start = -1
sma_end = -4
recent_swing_start = -3
recent_swing_end = -20
risk = 300

# one dict for one type, list to store all dicts?

all_dict = {'UR1':[],'DR1':[],
            'bull_fs':[],'bear_fs':[], 'bull_fs_sma':[], 'bear_fs_sma':[],
            'UC1': [], 'DC1': []
           }

# can consider to append df if match also, so no need to scrape again
# loop through each symbol
for i, ticker in enumerate(stock_list_all): # i is mainly for printing only
  #ticker = ticker.replace(".", "-")
  try:
    df = get_stock_price(ticker, freq = freq)

    bearish_ur_result, swing_bar_ur = bearish_ur1(df,day,rise_days=rise_drop_days)
    bullish_dr_result, swing_bar_dr = bullish_dr1(df,day,drop_days=rise_drop_days)
    bearish_fs_result, which_bar_bear_fs = bearish_fs(df,day)
    bullish_fs_result, which_bar_bull_fs = bullish_fs(df,day)
    #print(ticker)
    bullish_uc1_result, most_recent_low, most_recent_high = bullish_uc1(df, day, sma_start = sma_start, sma_end = sma_end, recent_swing_start = recent_swing_start, recent_swing_end = recent_swing_end)

    if bullish_uc1_result:
        print("######## Bullish UC1 Signal:", i, ticker)
        ticker_df = {'Ticker': ticker, 'Levels': [most_recent_low, most_recent_high], 'Direction': 'Long'}
        ticker_df['Prices Entry'] = get_enter_prices(df, ticker, direction =  ticker_df['Direction'], risk = risk, currency = 'USD', ratio = 2)
        all_dict['UC1'].append(ticker_df)

    bearish_dc1_result, most_recent_low, most_recent_high = bearish_dc1(df, day, sma_start = sma_start, sma_end = sma_end, recent_swing_start = recent_swing_start, recent_swing_end = recent_swing_end)


    if bearish_dc1_result:
        print("######## Bearish DC1 Signal:", i, ticker)
        ticker_df = {'Ticker': ticker, 'Levels': [most_recent_low, most_recent_high], 'Direction': 'Short'}
        ticker_df['Prices Entry'] = get_enter_prices(df, ticker, direction =  ticker_df['Direction'], risk = risk, currency = 'USD', ratio = 2)
        all_dict['DC1'].append(ticker_df)

    if bearish_ur_result:

        #ur.append(ticker)
        print("Bearish UR1 Signal:", i, ticker)
        print("Swing Bar:", swing_bar_ur)

        levels_low, levels_high = find_levels(df)

        ticker_df = {'Ticker': ticker,
                   'Levels': levels_high, 'Swing Bar': swing_bar_ur, 'Direction': 'Short'}

        # force_top, level = test_force_top(df.iloc[i-swing_bar:], levels_high)
        force_top, level = test_force_top(df, levels_high)

        if force_top:
            print("######## Force top:", level)
            ticker_df['Force Top'] = level
            ticker_df['Steepness'] = check_50_steepness_top(df, swing_bar_ur, level, rise_days = rise_drop_days)
            ticker_df['Prices Entry'] = get_enter_prices(df, ticker, direction =  ticker_df['Direction'], risk = risk, currency = 'USD', ratio = 2)
            all_dict['UR1'].append(ticker_df)
        elif force_top == False:
            print("No force top")
            ticker_df['Force Top'] = False

            #bearish_ur_dict[ticker] = dict_df

    if bullish_dr_result:
        #dict_df = {}
        #dr.append(ticker)
        print("Bullish DR1 Signal:", i, ticker)
        #print("Swing Bar:", swing_bar_dr)
        levels_low, levels_high = find_levels(df)

        ticker_df = {'Ticker': ticker,
                   'Levels': levels_low, 'Swing Bar': swing_bar_dr, 'Direction': 'Long'}

        force_bottom, level = test_force_bottom(df, levels_low)


        if force_bottom: # true if have, None if don't have
            print("######## Force bottom:", level)
            ticker_df['Force Bottom'] = level # level is tuple with datetime index and level itself
            ticker_df['Steepness'] = check_50_steepness_bottom(df, swing_bar_dr, level, drop_days = rise_drop_days)
            ticker_df['Prices Entry'] = get_enter_prices(df, ticker, direction =  ticker_df['Direction'], risk = risk, currency = 'USD', ratio = 2)
            all_dict['DR1'].append(ticker_df)
        elif force_bottom == False:
            print("No force bottom")
            ticker_df['Force Bottom'] = False





    if bearish_fs_result:

        #ur.append(ticker)
        print("Bearish FS Signal:", i, ticker)
        #print("Swing Bar:", swing_bar_ur)

        levels_low, levels_high = find_levels(df)
        below_sma = test_sma_below(df, -1,-6)
        #dr.append(ticker)
        print("Bullish FS Signal:", i, ticker)
        #print("Swing Bar:", swing_bar_dr)
        levels_low, levels_high = find_levels(df)

        ticker_df = {'Ticker': ticker,
                   'Levels': levels_low,
                     'FS Bar': which_bar_bull_fs, 'Direction': 'Long'}

        force_bottom, level = test_force_bottom(df, levels_low)
        above_sma = test_sma_above(df, -1, -6)

        if force_bottom:
            print("######## Force bottom:", level)
            ticker_df['Force Bottom'] = level
            ticker_df['Prices Entry'] = get_enter_prices(df, ticker, direction =  ticker_df['Direction'], risk = 300, currency = 'USD', ratio = 2)
            all_dict['bull_fs'].append(ticker_df)
        elif force_bottom == False:
            print("No force bottom")
            ticker_df['Force Bottom'] = False

        if above_sma:
            print("######## Above SMA Detected")
            ticker_df['Prices Entry'] = get_enter_prices(df, ticker, direction =  ticker_df['Direction'], risk = 300, currency = 'USD', ratio = 2)
            all_dict['bull_fs_sma'].append(ticker_df)
        elif above_sma == False:
            print("No above SMA")

    if bullish_fs_result:
        #dict_df = {}
        #dr.append(ticker)
        print("Bullish FS Signal:", i, ticker)
        #print("Swing Bar:", swing_bar_dr)
        levels_low, levels_high = find_levels(df)

        ticker_df = {'Ticker': ticker,
                   'Levels': levels_low,
                     'FS Bar': which_bar_bull_fs, 'Direction': 'Long'}

        force_bottom, level = test_force_bottom(df, levels_low)
        above_sma = test_sma_above(df, -1, -6)

        if force_bottom:
            print("######## Force bottom:", level)
            ticker_df['Force Bottom'] = level
            ticker_df['Prices Entry'] = get_enter_prices(df, ticker, direction =  ticker_df['Direction'], risk = 300, currency = 'USD', ratio = 2)
            all_dict['bull_fs'].append(ticker_df)
        elif force_bottom == False:
            print("No force bottom")
            ticker_df['Force Bottom'] = False

        if above_sma:
            print("######## Above SMA Detected")
            ticker_df['Prices Entry'] = get_enter_prices(df, ticker, direction =  ticker_df['Direction'], risk = 300, currency = 'USD', ratio = 2)
            all_dict['bull_fs_sma'].append(ticker_df)
        elif above_sma == False:
            print("No above SMA")

    if bullish_fs_result:
        #dict_df = {}
        #dr.append(ticker)
        print("Bullish FS Signal:", i, ticker)
        #print("Swing Bar:", swing_bar_dr)
        levels_low, levels_high = find_levels(df)

        ticker_df = {'Ticker': ticker,
                   'Levels': levels_low,
                     'FS Bar': which_bar_bull_fs, 'Direction': 'Long'}

        force_bottom, level = test_force_bottom(df, levels_low)
        above_sma = test_sma_above(df, -1, -6)

        if force_bottom:
            print("######## Force bottom:", level)
            ticker_df['Force Bottom'] = level
            ticker_df['Prices Entry'] = get_enter_prices(df, ticker, direction =  ticker_df['Direction'], risk = 300, currency = 'USD', ratio = 2)
            all_dict['bull_fs'].append(ticker_df)
        elif force_bottom == False:
            print("No force bottom")
            ticker_df['Force Bottom'] = False

        if above_sma:
            print("######## Above SMA Detected")
            ticker_df['Prices Entry'] = get_enter_prices(df, ticker, direction =  ticker_df['Direction'], risk = 300, currency = 'USD', ratio = 2)
            all_dict['bull_fs_sma'].append(ticker_df)
        elif above_sma == False:
            print("No above SMA")


  except Exception as e:
    print(e)

# # for visualization
# def plot_all(levels, df):
#   fig, ax = plt.subplots(figsize=(5, 2))
#   candlestick_ohlc(ax,df.values,width=0.6, colorup='green', colordown='red', alpha=1)
#   #ax = mpf.plot(df[-50:], type='candle', returnfig=True)
#   date_format = mpl_dates.DateFormatter('%d %b %Y')
#   ax.xaxis.set_major_formatter(date_format)
#   for level in levels:
#     ax.hlines(level[1], xmin = df['Date'][level[0]], xmax = max(df['Date']), colors='blue', linestyle='--')
#   fig.show()

#for visualization
def plot_all(levels, df, ticker, fs_bar = None):
    #fig = go.Figure(data=go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']),
                          go.Scatter(x=df.index, y=df.SMA50, line=dict(color='yellow', width=1)),
                          go.Scatter(x=df.index, y=df.SMA20, line=dict(color='blue', width=1))])


    for level in levels:
        fig.add_shape(type="line", x0=level[0], x1=df.index[-1], y0=level[1], y1 =level[1],
                      line_width=1, line_dash="dash", line_color="blue")
    if fs_bar:
        fig.add_shape(type="rect", x0=df.index[-fs_bar], x1=df.index[-1], y0=df['Low'][-fs_bar], y1=df['High'][-fs_bar],
                      line_width=1, line_dash="solid", line_color="yellow", fillcolor="yellow", opacity=0.3)

    fig.update_layout(title = ticker, height=800)
    #fig.update_layout(autosize=False, width=500, height=400) # for small
    fig.update_yaxes(fixedrange=False)
    fig.show()

# df = get_stock_price("BJ") # see why it is needed to ignore most recent swing
# levels_low, levels_high = find_levels(df)

# # test steepness calculation (this is correct)

# df = get_stock_price("BJ") # see why it is needed to ignore most recent swing
# bullish_dr_result, swing_bar_dr = bullish_dr1(df,day,drop_days=rise_drop_days)
# levels_low, levels_high = find_levels(df)
# force_top, level = test_force_bottom(df, levels_low)
# print(force_top, level)
# check_50_steepness_top(df, swing_bar_dr, level, rise_days = rise_drop_days)
# # plot_all(levels_low, df, "J", 50)


# start_of_swing = -swing_bar_dr - 3 - 3
#    # return (df['High'][start_of_swing] - level[1]) > 0.5*((df.loc[level[0]:df.index[start_of_swing]]['High'].max()) - level[1])
# df['High'][start_of_swing:-swing_bar_dr].max()



# (df['High'][start_of_swing:-swing_bar_dr].max() - level[1]) > 0.5*((df.loc[level[0]:df.index[start_of_swing]]['High'].max()) - level[1])

all_dict

# all_dict_serial = all_dict.copy()
# for k, ticker_dicts in all_dict_serial.items():
#     for ticker_dict in ticker_dicts:
#         if "Levels" in ticker_dict:
#             ticker_dict['Levels'] = [(str(l[0]), l[1]) for l in ticker_dict['Levels']]
#         if "Force Top" in ticker_dict:
#             if ticker_dict['Force Top']:
#                 ticker_dict['Force Top'] = (str(ticker_dict['Force Top'][0]), ticker_dict['Force Top'][1])
#         if "Force Bottom" in ticker_dict:
#             if ticker_dict['Force Bottom']:
#                 ticker_dict['Force Bottom'] = (str(ticker_dict['Force Bottom'][0]), ticker_dict['Force Bottom'][1])
#             #print(ticker_dict['Levels'])
# all_dict_serial



# # Store data (serialize)
with open('interested_tickers.pickle', 'wb') as handle:
    pickle.dump(all_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# # Load data (deserialize)
# with open('tickers.pickle', 'rb') as handle:
#     unserialized_data = pickle.load(handle)

flip_dict = {}
for strategy, ticker_dict_list in all_dict.items():
    for ticker_dict in ticker_dict_list:
        # print(strategy, ticker_dict['Ticker'])
        try:
            flip_dict[ticker_dict['Ticker']]
            flip_dict[ticker_dict['Ticker']].append(strategy)
        except:
            flip_dict[ticker_dict['Ticker']]= [strategy]


# # Store data (serialize)
with open('flip_dict.pickle', 'wb') as handle:
    pickle.dump(flip_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




# datetime object containing current date and time
now = datetime.datetime.now()
dt_string = now.strftime("%m/%d/%Y %I:%M:%S %p")
timezone_string = datetime.datetime.now().astimezone().tzname()
print(dt_string, timezone_string)

# exporting
import pprint
#for visualization
def plot_all_with_return(levels, df, ticker, fs_bar = None):
    #fig = go.Figure(data=go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']),
                          go.Scatter(x=df.index, y=df.SMA50, line=dict(color='yellow', width=1)),
                          go.Scatter(x=df.index, y=df.SMA20, line=dict(color='blue', width=1))])
    for level in levels:
        fig.add_shape(type="line", x0=level[0], x1=df.index[-1], y0=level[1], y1 =level[1],
                      line_width=1, line_dash="dash", line_color="blue")
    if fs_bar:
        fig.add_shape(type="rect", x0=df.index[-fs_bar], x1=df.index[-1], y0=df['Low'][-fs_bar], y1=df['High'][-fs_bar],
                      line_width=1, line_dash="solid", line_color="yellow", fillcolor="yellow", opacity=0.3)

    fig.update_layout(title = ticker, height=800)
    #fig.update_layout(autosize=False, width=500, height=400) # for small
    fig.update_yaxes(fixedrange=False)
    #fig.show()
    return fig

with open('interested_tickers.html', 'a') as f:
    f.truncate(0) # clear file if something is already written on it
    #title = "<h1>Tickers</h1>"
    #f.write(title)
    updated = "<h3>Last updated: <span id='timestring'></span></h3>"
    # GitHub Actions server timezone may not be at the same timezone of person opening the page on browser
    # hence Javascript code is written below to convert to client timezone before printing it on
    current_time = "<script>var date = new Date('" + dt_string + " " + timezone_string + "'); document.getElementById('timestring').innerHTML += date.toString()</script>"
    htmlLines = []
    for textLine in pprint.pformat(flip_dict).splitlines():
      htmlLines.append('<br/>%s' % textLine) # or something even nicer
    htmlText = '\n'.join(htmlLines)

    f.write(updated + current_time + htmlText)
    for strategy, ticker_dict_list in all_dict.items():
      f.write(f"<h2>{strategy}</h2>")
      for ticker_dict in ticker_dict_list:
          ticker = ticker_dict['Ticker']
          #if ticker in stock_list_snp:
          print(strategy, ticker)
          df = get_stock_price(ticker, freq = freq)
          ticker_dict['Volume'] = df['Volume'][-1]
          if (ticker_dict.get('FS Bar', None)):
              fig = plot_all_with_return(ticker_dict['Levels'],df,ticker + ': ' + strategy, fs_bar = ticker_dict['FS Bar'])
          else:
              fig = plot_all_with_return(ticker_dict['Levels'],df,ticker + ': ' + strategy)
          ticker_dict_no_prices = ticker_dict.copy()
          prices_dict = ticker_dict_no_prices.pop('Prices Entry')
          ticker_dict_no_prices.pop('Levels')
          htmlText2 = pd.DataFrame.from_dict(ticker_dict_no_prices, orient='index').to_html()
          htmlText3 = pd.DataFrame.from_dict(prices_dict, orient='index').to_html()
          f.write(htmlText2 + htmlText3)
          f.write(fig.to_html(full_html=False, include_plotlyjs='cdn')) # write the fig created above into the html fi


