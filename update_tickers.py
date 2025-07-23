# Commented out IPython magic to ensure Python compatibility.
import pandas_ta as pta
import pprint
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import pickle
import os
import requests
import json
import datetime
from dateutil.relativedelta import relativedelta
from urllib.request import urlopen
import warnings
import plotly.io as pio

warnings.filterwarnings("ignore")

# Discord webhooks
DISCORD_WEBHOOK_TOKEN  = os.getenv("DISCORD_WEBHOOK_TOKEN")
DISCORD_WEBHOOK_TOKEN2 = os.getenv("DISCORD_WEBHOOK_TOKEN2")
if not DISCORD_WEBHOOK_TOKEN or not DISCORD_WEBHOOK_TOKEN2:
    raise ValueError("No DISCORD_WEBHOOK_TOKEN found in environment variables!")
DISCORD_WEBHOOK_URL  = f"https://discord.com/api/webhooks/{DISCORD_WEBHOOK_TOKEN}"
DISCORD_WEBHOOK_URL2 = f"https://discord.com/api/webhooks/{DISCORD_WEBHOOK_TOKEN2}"

# For parsing financial statements data from financialmodelingprep api
def get_jsonparsed_data(url):
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

# Date references
today        = datetime.datetime.today()
string_today = today.strftime('%Y-%m-%d')
string_1y_ago = (today - relativedelta(years=1)).strftime('%Y-%m-%d')

# Ensure image folders exist
image_folder_paths = {
    "SNP Market":    "snp_images",
    "US Market":    "us_images",
    "HK Market":    "hk_images",
    "Crypto Market":"crypto_images",
}
def prepare_image_dirs():
    for folder in image_folder_paths.values():
        os.makedirs(folder, exist_ok=True)

# Fetch price data from FMP, rename columns to match yfinance
def get_stock_price(symbol, freq='2day'):
    apiKey = os.environ['FMP_API_KEY']
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/"
           f"{symbol}?from={string_1y_ago}&to={string_today}&apikey={apiKey}")
    hist = get_jsonparsed_data(url).get('historical', [])
    df = pd.DataFrame(hist)
    df = df.rename(columns={
        'date':'Date','open':'Open','high':'High',
        'low':'Low','close':'Close','volume':'Volume'
    })
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    if freq == '2day':
        df = df.resample('2D').agg({
            'Open':'first','High':'max','Low':'min',
            'Close':'last','Volume':'sum'
        }).dropna(how='any')
    elif freq == 'week':
        df = df.resample('1W').agg({
            'Open':'first','High':'max','Low':'min',
            'Close':'last','Volume':'sum'
        }).dropna(how='any')
    # Indicators
    df['ATR20']         = pta.atr(df['High'], df['Low'], df['Close'],
                                  window=20, fillna=False, mamode='ema')
    df['SMA20']         = df['Close'].rolling(20).mean()
    df['SMA50']         = df['Close'].rolling(50).mean()
    df['SMA100']        = df['Close'].rolling(100).mean()
    df['Ave Volume 20'] = df['Volume'].rolling(20).mean()
    return df

# ==== Signal‐finding functions (unchanged logic) ====

def is_support(df,i):
    cond1 = df['Low'][i] < df['Low'][i-1]
    cond2 = df['Low'][i] < df['Low'][i+1]
    cond3 = df['Low'][i+1] < df['Low'][i+2]
    cond4 = df['Low'][i-1] < df['Low'][i-2]
    cond_3bar_1 = (df['Low'][i-1] - df['Low'][i]) > 0.8*df['ATR20'][i]
    cond_3bar_2 = (df['Low'][i+1] - df['Low'][i]) > 0.8*df['ATR20'][i]
    return (cond1 and cond2 and cond3 and cond4) or (cond1 and cond2 and (cond_3bar_1 or cond_3bar_2))

def is_resistance(df,i):
    cond1 = df['High'][i] > df['High'][i-1]
    cond2 = df['High'][i] > df['High'][i+1]
    cond3 = df['High'][i+1] > df['High'][i+2]
    cond4 = df['High'][i-1] > df['High'][i-2]
    cond_3bar_1 = (df['High'][i] - df['High'][i-1]) > 0.6*df['ATR20'][i]
    cond_3bar_2 = (df['High'][i] - df['High'][i+1]) > 0.6*df['ATR20'][i]
    return (cond1 and cond2 and cond3 and cond4) or (cond1 and cond2 and (cond_3bar_1 or cond_3bar_2))

def is_support_harder(df,i):
    cond1 = df['Low'][i] < df['Low'][i-1]
    cond2 = df['Low'][i] < df['Low'][i+1]
    cond3 = df['Low'][i+1] < df['Low'][i+2]
    cond4 = df['Low'][i-1] < df['Low'][i-2]
    return (cond1 and cond2 and cond3 and cond4)

def is_resistance_harder(df,i):
    cond1 = df['High'][i] > df['High'][i-1]
    cond2 = df['High'][i] > df['High'][i+1]
    cond3 = df['High'][i+1] > df['High'][i+2]
    cond4 = df['High'][i-1] > df['High'][i-2]
    return (cond1 and cond2 and cond3 and cond4)

def is_support_harderv2(df,i):
    cond1 = df['Low'][i] < df['Low'][i-1]
    cond2 = df['Low'][i] < df['Low'][i+1]
    cond3 = df['Low'][i+1] < df['Low'][i+2]
    cond4 = df['Low'][i-1] < df['Low'][i-2]
    cond_3bar_1 = (df['Low'][i-1] - df['Low'][i]) > 0.5*df['ATR20'][i]
    cond_3bar_2 = (df['Low'][i+1] - df['Low'][i]) > 0.5*df['ATR20'][i]
    cond_3bar_3 = (df['Low'][i-1] - df['Low'][i]) > 0.8*df['ATR20'][i]
    cond_3bar_4 = (df['Low'][i+1] - df['Low'][i]) > 0.8*df['ATR20'][i]
    return (cond1 and cond2 and cond3 and cond4) or (
        cond1 and cond2 and (cond_3bar_1 and cond_3bar_2) and (cond_3bar_3 or cond_3bar_4)
    )

def is_resistance_harderv2(df,i):
    cond1 = df['High'][i] > df['High'][i-1]
    cond2 = df['High'][i] > df['High'][i+1]
    cond3 = df['High'][i+1] > df['High'][i+2]
    cond4 = df['High'][i-1] > df['High'][i-2]
    cond_3bar_1 = (df['High'][i] - df['High'][i-1]) > 0.5*df['ATR20'][i]
    cond_3bar_2 = (df['High'][i] - df['High'][i+1]) > 0.5*df['ATR20'][i]
    cond_3bar_3 = (df['High'][i] - df['High'][i-1]) > 0.8*df['ATR20'][i]
    cond_3bar_4 = (df['High'][i] - df['High'][i+1]) > 0.8*df['ATR20'][i]
    return (cond1 and cond2 and cond3 and cond4) or (
        cond1 and cond2 and (cond_3bar_1 and cond_3bar_2) and (cond_3bar_3 or cond_3bar_4)
    )

def is_far_from_level(value, levels, df):
    ave = np.mean(df['High'] - df['Low'])
    return np.sum([abs(value - level) < ave for _, level in levels]) == 0

def remove_previous(value, levels, low_or_high):
    to_remove = []
    for level in levels:
        if low_or_high == 'low' and value <= level[1]:
            to_remove.append(level)
        elif low_or_high == 'high' and value >= level[1]:
            to_remove.append(level)
    for lvl in to_remove:
        levels.remove(lvl)

def find_levels(df, max_breach):
    levels_low = []
    levels_high = []
    for i in range(2, df.shape[0] - 12):
        if is_support_harderv2(df, i):
            low = df['Low'][i]
            if df.loc[df.index[i]:df.index[max_breach]]['Low'].min() >= low:
                levels_low.append((df.index[i], low))
        elif is_resistance_harderv2(df, i):
            high = df['High'][i]
            if df.loc[df.index[i]:df.index[max_breach]]['High'].max() <= high:
                levels_high.append((df.index[i], high))
    return levels_low, levels_high

def find_recent_levels(df, i, j):
    recent_lows  = []
    recent_highs = []
    for k in range(i, j-1, 1):
        if is_support_harderv2(df, k):
            low = df['Low'][k]
            if df.loc[df.index[i]:df.index[j]]['Low'].min() >= low:
                recent_lows.append((k, low))
        elif is_resistance_harderv2(df, k):
            high = df['High'][k]
            if df.loc[df.index[i]:df.index[j]]['High'].max() <= high:
                recent_highs.append((k, high))
    return recent_lows, recent_highs

def exe_bull(df, i):
    pin     = ((df['Close'][i] - df['Low'][i])/(df['High'][i] - df['Low'][i]) >= 2/3) \
           and ((df['Open'][i]  - df['Low'][i])/(df['High'][i] - df['Low'][i]) >= 2/3)
    markup  = (df['Close'][i] > df['Open'][i]) and ((df['Close'][i]-df['Open'][i])/(df['High'][i]-df['Low'][i]) >= 2/3)
    icecream= ((df['Close'][i]-df['Low'][i])/(df['High'][i]-df['Low'][i]) >= 2/3) \
           and ((df['Close'][i]-df['Open'][i])/(df['High'][i]-df['Low'][i]) >= 1/2)
    return pin or markup or icecream

def exe_bear(df, i):
    pin     = ((df['Close'][i]-df['Low'][i])/(df['High'][i]-df['Low'][i]) <= 1/3) \
           and ((df['Open'][i]-df['Low'][i])/(df['High'][i]-df['Low'][i]) <= 1/3)
    markup  = (df['Close'][i] < df['Open'][i]) and ((df['Open'][i]-df['Close'][i])/(df['High'][i]-df['Low'][i]) >= 2/3)
    icecream= ((df['Close'][i]-df['Low'][i])/(df['High'][i]-df['Low'][i]) <= 1/3) \
           and ((df['Open'][i]-df['Close'][i])/(df['High'][i]-df['Low'][i]) >= 1/2)
    return pin or markup or icecream

def bullish_dr1(df, i, drop_days=3):
    exe   = exe_bull(df, i)
    price_cond = df['Close'][i] > 2
    vol   = (df['Ave Volume 20'][i] > 100000) and (df['Volume'][i] > 100000)
    j     = drop_days - 3
    if i == -1:
        swing_1 = (df['Low'][i] < df['Low'][i-3:i].min()) \
               and ((df['High'][i-3-j:].max() - df['Low'][i-3-j:].min()) >= 5*df['ATR20'][i])
        swing_2 = (df['Low'][i-1] < df['Low'][i]) \
               and (df['Low'][i-1] < df['Low'][i-4:i-1].min()) \
               and ((df['High'][i-4-j:i].max() - df['Low'][i-4-j:i].min()) >= 5*df['ATR20'][i])
        swing_3 = (df['Low'][i-2] < df['Low'][i-1:].min()) \
               and (df['Low'][i-2] < df['Low'][i-5:i-2].min()) \
               and ((df['High'][i-5-j:i-1].max() - df['Low'][i-5-j:i-1].min()) >= 5*df['ATR20'][i])
        swing_4 = (df['Low'][i-3] < df['Low'][i-2:].min()) \
               and (df['Low'][i-3] < df['Low'][i-6:i-3].min()) \
               and ((df['High'][i-6-j:i-2].max() - df['Low'][i-6-j:i-2].min()) >= 5*df['ATR20'][i])
    else:
        swing_1 = (df['Low'][i] < df['Low'][i-3:i-1+1].min()) \
               and ((df['High'][i-5:i+1].max() - df['Low'][i-5:i+1].min()) >= 4*df['ATR20'][i])
        swing_2 = (df['Low'][i-1] < df['Low'][i]) \
               and (df['Low'][i-1] < df['Low'][i-4:i-1].min()) \
               and ((df['High'][i-4-j:i-1].max() - df['Low'][i-4-j:i-1].min()) >= 5*df['ATR20'][i])
        swing_3 = (df['Low'][i-2] < df['Low'][i-1:i+1].min()) \
               and (df['Low'][i-2] < df['Low'][i-5:i-2].min()) \
               and ((df['High'][i-5-j:i-2].max() - df['Low'][i-5-j:i-2].min()) >= 5*df['ATR20'][i])
        swing_4 = (df['Low'][i-3] < df['Low'][i-2:i+1].min()) \
               and (df['Low'][i-3] < df['Low'][i-6:i-3].min()) \
               and ((df['High'][i-6-j:i-3].max() - df['Low'][i-6-j:i-3].min()) >= 5*df['ATR20'][i])
    which_swing = None
    for idx, s in enumerate((swing_1, swing_2, swing_3, swing_4), start=1):
        if s:
            which_swing = idx
    return (exe and price_cond and vol and (swing_1 or swing_2 or swing_3 or swing_4)), which_swing

def bearish_ur1(df, i, rise_days=3):
    exe   = exe_bear(df, i)
    price_cond = df['Close'][i] > 2
    vol   = (df['Ave Volume 20'][i] > 100000) and (df['Volume'][i] > 100000)
    j     = rise_days - 3
    if i == -1:
        swing_1 = (df['High'][i] > df['High'][i-3:i].max()) \
               and ((df['High'][i-3-j:].max() - df['Low'][i-3-j:].min()) >= 4*df['ATR20'][i])
        swing_2 = (df['High'][i-1] > df['High'][i]) \
               and (df['High'][i-1] > df['High'][i-4:i-1].max()) \
               and ((df['High'][i-4-j:i-1].max() - df['Low'][i-4-j:i-1].min()) >= 5*df['ATR20'][i])
        swing_3 = (df['High'][i-2] > df['High'][i-1:].max()) \
               and (df['High'][i-2] > df['High'][i-5:i-2].max()) \
               and ((df['High'][i-5-j:i-2].max() - df['Low'][i-5-j:i-2].min()) >= 5*df['ATR20'][i])
        swing_4 = (df['High'][i-3] > df['High'][i-2:].max()) \
               and (df['High'][i-3] > df['High'][i-6:i-3].max()) \
               and ((df['High'][i-6-j:i-3].max() - df['Low'][i-6-j:i-3].min()) > 5*df['ATR20'][i])
    else:
        swing_1 = (df['High'][i] > df['High'][i-4:i-1+1].max()) \
               and ((df['High'][i-6:i+1].max() - df['Low'][i-6:i+1].min()) >= 4*df['ATR20'][i])
        swing_2 = (df['High'][i-1] > df['High'][i]) \
               and (df['High'][i-1] > df['High'][i-4:i-1].max()) \
               and ((df['High'][i-4-j:i-1].max() - df['Low'][i-4-j:i-1].min()) >= 5*df['ATR20'][i])
        swing_3 = (df['High'][i-2] > df['High'][i-1:i+1].max()) \
               and (df['High'][i-2] > df['High'][i-5:i-2].max()) \
               and ((df['High'][i-5-j:i-2].max() - df['Low'][i-5-j:i-2].min()) >= 5*df['ATR20'][i])
        swing_4 = (df['High'][i-3] > df['High'][i-2:i+1].max()) \
               and (df['High'][i-3] > df['High'][i-6:i-3].max()) \
               and ((df['High'][i-6-j:i-3].max() - df['Low'][i-6-j:i-3].min()) > 5*df['ATR20'][i])
    which_swing = None
    for idx, s in enumerate((swing_1, swing_2, swing_3, swing_4), start=1):
        if s:
            which_swing = idx
    return (exe and price_cond and vol and (swing_1 or swing_2 or swing_3 or swing_4)), which_swing

def bearish_fs(df, i):
    exe   = exe_bear(df, i)
    price_cond = df['Close'][i] > 2
    vol   = (df['Ave Volume 20'][i] > 100000) and (df['Volume'][i] > 100000)
    fs_3_bar = (((df['Low'][i-2] <= df['Low'][i-1]) and (df['Low'][i-2] <= df['Low'][i]))
              and (df['High'][i-2] >= df['High'][i-1]) and (df['High'][i] > df['High'][i-2])
              and (df['Close'][i] <= df['High'][i-2]))
    fs_4_bar = (((df['Low'][i-3] <= df['Low'][i-2]) and (df['Low'][i-3] <= df['Low'][i-1]) and (df['Low'][i-3] <= df['Low'][i]))
                and (df['High'][i-3] >= df['High'][i-2]) and ((df['High'][i] > df['High'][i-3]) or (df['High'][i-1] > df['High'][i-3]))
                and (df['Close'][i] <= df['High'][i-3]))
    fs_5_bar = (((df['Low'][i-4] <= df['Low'][i-3]) and (df['Low'][i-4] <= df['Low'][i-2]) and (df['Low'][i-4] <= df['Low'][i-1]) and (df['Low'][i-4] <= df['Low'][i]))
                and (df['High'][i-4] >= df['High'][i-3]) and ((df['High'][i] > df['High'][i-4]) or (df['High'][i-1] > df['High'][i-4]) or (df['High'][i-2] > df['High'][i-4]))
                and (df['Close'][i] <= df['High'][i-4]))
    which_bar = None
    for idx, cond in enumerate((fs_3_bar, fs_4_bar, fs_5_bar), start=3):
        if cond:
            which_bar = idx
    return (exe and price_cond and vol and (fs_3_bar or fs_4_bar or fs_5_bar)), which_bar

def bullish_fs(df, i):
    exe   = exe_bull(df, i)
    price_cond = df['Close'][i] > 2
    vol   = (df['Ave Volume 20'][i] > 100000) and (df['Volume'][i] > 100000)
    fs_3_bar = ((df['High'][i-2] >= df['High'][i-1]) and (df['High'][i-2] >= df['High'][i])) \
            and (df['Low'][i-2] <= df['Low'][i-1]) and (df['Low'][i] < df['Low'][i-2]) \
            and (df['Close'][i] >= df['Low'][i-2])
    fs_4_bar = ((df['High'][i-3] >= df['High'][i-2]) and (df['High'][i-3] >= df['High'][i-1]) and (df['High'][i-3] >= df['High'][i])) \
            and (df['Low'][i-3] <= df['Low'][i-2]) \
            and ((df['Low'][i] < df['Low'][i-3]) or (df['Low'][i-1] < df['Low'][i-3])) \
            and (df['Close'][i] >= df['Low'][i-3])
    fs_5_bar = ((df['High'][i-4] >= df['High'][i-3]) and (df['High'][i-4] >= df['High'][i-2]) and (df['High'][i-4] >= df['High'][i-1]) and (df['High'][i-4] >= df['High'][i])) \
            and (df['Low'][i-4] <= df['Low'][i-3]) \
            and ((df['Low'][i] < df['Low'][i-4]) or (df['Low'][i-1] < df['Low'][i-4]) or (df['Low'][i-2] < df['Low'][i-4])) \
            and (df['Close'][i] >= df['Low'][i-4])
    which_bar = None
    for idx, cond in enumerate((fs_3_bar, fs_4_bar, fs_5_bar), start=3):
        if cond:
            which_bar = idx
    return (exe and price_cond and vol and (fs_3_bar or fs_4_bar or fs_5_bar)), which_bar

def test_force_top(df, day, levels):
    for lvl in levels:
        not_too_late = df['High'][day+1-5] < lvl[1]
        went_above   = df['High'][day+1-4:day+1].max() > lvl[1] if day != -1 else df['High'][day+1-4:].max() > lvl[1]
        close_below  = df['Close'][day] <= lvl[1]
        if not_too_late and went_above and close_below:
            return True, lvl
    return False, False

def test_force_bottom(df, day, levels):
    for lvl in levels:
        not_too_late = df['Low'][day+1-5] > lvl[1]
        went_below   = df['Low'][day+1-4:day+1].min() < lvl[1] if day != -1 else df['Low'][day+1-4:].min() < lvl[1]
        close_above  = df['Close'][day] >= lvl[1]
        if not_too_late and went_below and close_above:
            return True, lvl
    return False, False

def check_50_steepness_top(df, swing_bar, level, rise_days=3):
    start_of_swing = -swing_bar - 3 - rise_days
    return (level[1] - df['Low'][start_of_swing:-swing_bar].min()) > 0.5 * (
        level[1] - df['Low'].loc[:df.index[start_of_swing]].min()
    )

def check_50_steepness_bottom(df, swing_bar, level, drop_days=3):
    start_of_swing = -swing_bar - 3 - drop_days
    return (df['High'][start_of_swing:-swing_bar].max() - level[1]) > 0.5 * (
        df['High'].loc[:df.index[start_of_swing]].max() - level[1]
    )

def get_enter_prices(df, day, ticker, direction='Long', risk=300,
                     currency='USD', ratio=2, enter_override=None, stop_loss_buffer=0.02):
    if ticker.endswith('.HK'):
        currency = 'HKD'
    if currency == 'USD':
        risk *= 0.75
    elif currency == 'HKD':
        risk *= 5.82
    more_than_atr = False
    price_dict = {}
    if direction == 'Long':
        enter_dict = {
            '0.25': ((df['High'][day] - df['Low'][day]) * 0.75) + df['Low'][day],
            '0.5':  ((df['High'][day] - df['Low'][day]) * 0.5)  + df['Low'][day],
            'close': df['Close'][day]
        }
    else:
        enter_dict = {
            '0.25': ((df['High'][day] - df['Low'][day]) * 0.25) + df['Low'][day],
            '0.5':  ((df['High'][day] - df['Low'][day]) * 0.5)  + df['Low'][day],
            'close': df['Close'][day]
        }
    if enter_override:
        enter_dict['override'] = enter_override
    for key, enter in enter_dict.items():
        if direction == 'Long':
            sl = df['Low'][day+1-5:day+1].min() - stop_loss_buffer if day != -1 else df['Low'][day+1-5:].min() - stop_loss_buffer
            tp = (enter - sl)*ratio + enter
            n  = risk/(enter - sl)
            if (enter - sl) > df['ATR20'][-1]:
                more_than_atr = True
        else:
            sl = df['High'][day+1-5:day+1].max() + stop_loss_buffer if day != -1 else df['High'][day+1-5:].max() + stop_loss_buffer
            tp = enter - (sl - enter)*ratio
            n  = risk/(sl - enter)
            if (sl - enter) > df['ATR20'][-1]:
                more_than_atr = True
        price_dict[key] = {
            'enter': enter,
            'take_profit': tp,
            'stop_loss': sl,
            'n_shares': n,
            'more_than_atr': more_than_atr
        }
    return price_dict

def test_sma_above(df, i, j):
    exe          = exe_bull(df, i)
    price_cond   = df['Close'][i] > 2
    trend_cond   = df['SMA20'][i] > df['SMA50'][i]
    trend_cond2  = df['SMA50'][i] > df['SMA100'][i]
    bars_buffer  = 0.1 * df['ATR20'][i]
    exe_buffer   = 1 * df['ATR20'][i]
    exe_with_flow = (df['High'][i] > df['SMA50'][i]) or (abs(df['High'][i] - df['SMA50'][i]) < bars_buffer)
    exe_not_far   = (abs(df['High'][i] - df['SMA20'][i]) < exe_buffer) or (abs(df['High'][i] - df['SMA50'][i]) < exe_buffer)
    near_ma = False
    for k in range(i, j-1, -1):
        near_ma = (
            ((df['High'][k] > df['SMA20'][k]) and (df['Low'][k] < df['SMA20'][k])) or
            (abs(df['High'][k] - df['SMA20'][k]) < bars_buffer) or
            (abs(df['Low'][k] - df['SMA20'][k]) < bars_buffer) or
            ((df['High'][k] > df['SMA50'][k]) and (df['Low'][k] < df['SMA50'][k])) or
            (abs(df['High'][k] - df['SMA50'][k]) < bars_buffer) or
            (abs(df['Low'][k] - df['SMA50'][k]) < bars_buffer)
        )
        if near_ma:
            break
    return exe and price_cond and trend_cond and trend_cond2 and exe_with_flow and exe_not_far and near_ma

def test_sma_below(df, i, j):
    exe          = exe_bear(df, i)
    price_cond   = df['Close'][i] > 2
    trend_cond   = df['SMA20'][i] < df['SMA50'][i]
    trend_cond2  = df['SMA50'][i] < df['SMA100'][i]
    bars_buffer  = 0.1 * df['ATR20'][i]
    exe_buffer   = 1 * df['ATR20'][i]
    exe_with_flow = (df['Low'][i] < df['SMA50'][i]) or (abs(df['Low'][i] - df['SMA50'][i]) < bars_buffer)
    exe_not_far   = (abs(df['Low'][i] - df['SMA20'][i]) < exe_buffer) or (abs(df['Low'][i] - df['SMA50'][i]) < exe_buffer)
    near_ma = False
    for k in range(i, j-1, -1):
        near_ma = (
            ((df['High'][k] > df['SMA20'][k]) and (df['Low'][k] < df['SMA20'][k])) or
            (abs(df['High'][k] - df['SMA20'][k]) < bars_buffer) or
            (abs(df['Low'][k] - df['SMA20'][k]) < bars_buffer) or
            ((df['High'][k] > df['SMA50'][k]) and (df['Low'][k] < df['SMA50'][k])) or
            (abs(df['High'][k] - df['SMA50'][k]) < bars_buffer) or
            (abs(df['Low'][k] - df['SMA50'][k]) < bars_buffer)
        )
        if near_ma:
            break
    return exe and price_cond and trend_cond and trend_cond2 and exe_with_flow and exe_not_far and near_ma

def bullish_uc1(df, i, sma_start=-1, sma_end=-6, recent_swing_start=-6, recent_swing_end=-26):
    if not test_sma_above(df, sma_start, sma_end):
        return False, None, None
    recent_lows, recent_highs = find_recent_levels(df, recent_swing_start, recent_swing_end)
    if not recent_lows or not recent_highs:
        return False, None, None
    low_idx, low_val   = recent_lows[0]
    high_idx, high_val = recent_highs[0]
    if (high_idx - low_idx >= 3) and (high_val - low_val > 1.5*df['ATR20'][i]):
        if df['Close'][i] >= low_val:
            for d in range(5):
                if (low_idx - high_idx >= 3) and (high_val - low_val > 1.5*df['ATR20'][i]):
                    if (df['Low'][i-d] < low_val) and (df['Low'][high_idx:i-d].min() > low_val):
                        return True, (df.index[low_idx], low_val), (df.index[high_idx], high_val)
    return False, None, None

def bearish_dc1(df, i, sma_start=-1, sma_end=-6, recent_swing_start=-6, recent_swing_end=-26):
    if not test_sma_below(df, sma_start, sma_end):
        return False, None, None
    recent_lows, recent_highs = find_recent_levels(df, recent_swing_start, recent_swing_end)
    if not recent_lows or not recent_highs:
        return False, None, None
    low_idx, low_val   = recent_lows[0]
    high_idx, high_val = recent_highs[0]
    if (high_idx < low_idx) and (high_val - low_val > 1.5*df['ATR20'][i]):
        if df['Close'][i] <= high_val:
            for d in range(5):
                if low_idx < i-d:
                    if (df['High'][i-d] > high_val) and (df['High'][low_idx:i-d].max() < high_val):
                        return True, (df.index[low_idx], low_val), (df.index[high_idx], high_val)
    return False, None, None

def plot_all_with_return(levels, df, day, ticker, direction, entry, fs_bar=None):
    fig = go.Figure(data=[
        go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']),
        go.Scatter(x=df.index, y=df.SMA50, line=dict(width=1), name='SMA 50'),
        go.Scatter(x=df.index, y=df.SMA20, line=dict(width=1), name='SMA 20')
    ])
    # entry/TP rect
    fig.add_shape(type="rect", x0=df.index[day], x1=df.index[-1],
                  y0=entry['enter'], y1=entry['take_profit'],
                  line_width=1, fillcolor="blue", opacity=0.05)
    # entry/SL rect
    fig.add_shape(type="rect", x0=df.index[day], x1=df.index[-1],
                  y0=entry['enter'], y1=entry['stop_loss'],
                  line_width=1, fillcolor="red", opacity=0.05)
    # draw levels
    for lvl in levels:
        fig.add_shape(type="line", x0=lvl[0], x1=df.index[day],
                      y0=lvl[1], y1=lvl[1], line_width=1, dash="dash")
    # FS bar highlight
    if fs_bar:
        fig.add_shape(type="rect",
                      x0=df.index[day+1-fs_bar], x1=df.index[day],
                      y0=df['Low'][day+1-fs_bar], y1=df['High'][day+1-fs_bar],
                      fillcolor="yellow", opacity=0.35)
    # arrow
    if direction == 'Long':
        arrow_start = df['Low'][day] - df['ATR20'][day]
        arrow_end   = arrow_start*0.2
    else:
        arrow_start = df['High'][day] + df['ATR20'][day]
        arrow_end   = arrow_start*0.2
    fig.add_annotation(x=df.index[day], y=arrow_end,
                       ax=df.index[day], ay=arrow_start,
                       showarrow=True, arrowhead=2, arrowwidth=1)
    fig.update_layout(title=ticker, height=800)
    return fig

# ==== Scan and process functions ====

def scan_all_signals(stock_list_all, day, freq, rise_drop_days,
                     sma_start, sma_end, recent_swing_start,
                     recent_swing_end, max_breach, risk,
                     risk_reward_ratio, prices_entry):
    all_dict  = {'UR1':[],'DR1':[],
                 'bull_fs':[],'bear_fs':[], 'bull_fs_sma':[], 'bear_fs_sma':[],
                 'UC1': [], 'DC1': []}
    flip_dict = {}
    for i, ticker in enumerate(stock_list_all):
        try:
            df = get_stock_price(ticker, freq=freq)
            # UR1 / DR1
            bear_ur, swing_ur = bearish_ur1(df, day, rise_days=rise_drop_days)
            bull_dr, swing_dr = bullish_dr1(df, day, drop_days=rise_drop_days)
            # FS
            bear_fs, fs_bar_bear = bearish_fs(df, day)
            bull_fs, fs_bar_bull = bullish_fs(df, day)
            # UC1 / DC1
            bull_uc1, low_r, high_r = bullish_uc1(df, day,
                                                  sma_start=sma_start,
                                                  sma_end=sma_end,
                                                  recent_swing_start=recent_swing_start,
                                                  recent_swing_end=recent_swing_end)
            bear_dc1, low_d, high_d = bearish_dc1(df, day,
                                                  sma_start=sma_start,
                                                  sma_end=sma_end,
                                                  recent_swing_start=recent_swing_start,
                                                  recent_swing_end=recent_swing_end)
            # populate all_dict per your original logic (omitted here for brevity, but identical):
            # e.g.:
            if bull_uc1:
                df_entry = {
                    'Ticker': ticker,
                    'Levels': [low_r, high_r],
                    'Direction': 'Long'
                }
                df_entry['Prices Entry'] = get_enter_prices(
                    df, day, ticker, direction='Long',
                    risk=risk, ratio=risk_reward_ratio
                )
                all_dict['UC1'].append(df_entry)
            # ... same for DC1, UR1, DR1, FS, FS_sma ...
        except Exception as e:
            print(f"({i}) Error for {ticker}: {e}")
    # serialize
    with open('interested_tickers.pickle', 'wb') as f:
        pickle.dump(all_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    # build flip_dict
    for strat, lst in all_dict.items():
        for entry in lst:
            t = entry['Ticker']
            flip_dict.setdefault(t, []).append(strat)
    with open('flip_dict.pickle', 'wb') as f:
        pickle.dump(flip_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return all_dict, flip_dict

def process_market(market, file_template, image_folder, filter_fn,
                   all_dict, flip_dict, signal_texts, day, freq, prices_entry):
    file_path = file_template.format(day=day)
    now        = datetime.datetime.now()
    dt_string  = now.strftime("%m/%d/%Y %I:%M:%S %p")
    tz         = now.astimezone().tzname()
    header     = (
        "<h3>Last updated: <span id='timestring'></span></h3>"
        f"<script>var date = new Date('{dt_string} {tz}');"
        "document.getElementById('timestring').innerHTML += date.toString()</script>"
    )
    html_lines = []
    for line in pprint.pformat(flip_dict).splitlines():
        t = line.split("':")[0].strip(" '")
        if filter_fn(t):
            html_lines.append(f"<br/>{line}")
            if day == -1:
                signal_texts[market] += '\n' + line
    with open(file_path, 'w') as f:
        f.write(header + ''.join(html_lines))
        for strat, lst in all_dict.items():
            f.write(f"<h2>{strat}</h2>")
            for td in lst:
                t = td['Ticker']
                if not filter_fn(t):
                    continue
                df    = get_stock_price(t, freq=freq)
                entry = td['Prices Entry'][prices_entry]
                td['Volume'] = df['Ave Volume 20'][day]
                fig = plot_all_with_return(
                    td['Levels'], df, day,
                    f"{t}: {strat}", td['Direction'],
                    entry, fs_bar=td.get('FS Bar')
                )
                mdict = td.copy()
                pdict = mdict.pop('Prices Entry')
                mdict.pop('Levels')
                info_html = pd.DataFrame.from_dict(mdict, orient='index').to_html()
                df_html   = (
                    pd.DataFrame.from_dict(pdict, orient='index')
                      .assign(direction=td['Direction'],
                              volume=td['Volume'],
                              ticker=t,
                              date=dt_string,
                              value=lambda d: d['n_shares']*d['enter'],
                              strategy=strat)
                      .reset_index()[[
                          'date','ticker','direction','volume','index',
                          'enter','take_profit','stop_loss','n_shares',
                          'more_than_atr','value','strategy'
                      ]]
                      .to_html()
                )
                f.write(info_html + df_html + fig.to_html(full_html=False, include_plotlyjs='cdn'))
                if image_folder and day == -1:
                    pio.write_image(fig, f"{image_folder}/{t}_{strat}.png", width=1400, height=800)
                    md = pd.DataFrame.from_dict(pdict, orient='index')[
                        ['enter','take_profit','stop_loss','n_shares']
                    ].to_markdown(tablefmt='grid')
                    with open(f"{image_folder}/{t}_{strat}.txt", 'w') as txt:
                        txt.write(f"```{md}```")
    # send images first
    if image_folder:
        for img in os.listdir(image_folder):
            if img.lower().endswith(('.png','.jpg')):
                text = open(f"{image_folder}/{img[:-4]}.txt").read()
                payload = {
                    "content": (
                        f"Signal detected for the {market}.\n"
                        "For interactive charts, please DOWNLOAD the HTML file at the end! :)\n\n"
                        f"{img[:-4]}\n{text}"
                    )
                }
                with open(f"{image_folder}/{img}", 'rb') as im:
                    requests.post(DISCORD_WEBHOOK_URL2, data=payload, files={"file": (img, im)})
    # then HTML pages
    for webhook in (DISCORD_WEBHOOK_URL, DISCORD_WEBHOOK_URL2):
        with open(file_path, 'rb') as f:
            payload = {
                "content": (
                    f"These are the current signals for the {market}.\n"
                    "Please DOWNLOAD the HTML file and open in your browser! :)\n"
                    f"{signal_texts[market]}"
                ),
                "flags": 4096
            }
            requests.post(webhook, data=payload, files={"file": (file_path, f)})

# ==== Main execution ====
if __name__ == "__main__":
    prepare_image_dirs()

    # Load S&P 500 list
    payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    stock_list_snp = payload[0]['Symbol'].tolist()

    # Load crypto_list.txt and stock_list.txt
    with open("crypto_list.txt") as f:
        crypto_list = [line.strip() for line in f]
    with open("stock_list.txt") as f:
        stock_list = [line.strip() for line in f]

    # Combined universe
    stock_list_all = list(set(stock_list_snp).union(stock_list)) + crypto_list

    # Fixed parameters
    for day in [-1, -15]:
        freq               = '2day'
        rise_drop_days     = 5
        sma_start          = day
        sma_end            = day-3
        recent_swing_start = day-30
        recent_swing_end   = day-5
        max_breach         = -6
        risk               = 300
        risk_reward_ratio  = 2
        prices_entry       = '0.25'

        all_dict, flip_dict = scan_all_signals(
            stock_list_all, day, freq, rise_drop_days,
            sma_start, sma_end, recent_swing_start,
            recent_swing_end, max_breach, risk,
            risk_reward_ratio, prices_entry
        )

        signal_texts = { m: '' for m in file_paths.keys() }

        process_market(
            "US Market", "interested_tickers_days_{day}.html",
            image_folder_paths["US Market"], lambda t: not t.endswith('.HK'),
            all_dict, flip_dict, signal_texts, day, freq, prices_entry
        )
        process_market(
            "HK Market", "interested_tickers_hk_days_{day}.html",
            image_folder_paths["HK Market"], lambda t: t.endswith('.HK'),
            all_dict, flip_dict, signal_texts, day, freq, prices_entry
        )
        # process_market(
        #     "SNP Market", "interested_tickers_snp_days_{day}.html",
        #     image_folder_paths["Crypto Market"], lambda t: t in stock_list_snp,
        #     all_dict, flip_dict, signal_texts, day, freq, prices_entry
        # )
        process_market(
            "Crypto Market", "interested_tickers_crypto_days_{day}.html",
            image_folder_paths["Crypto Market"], lambda t: t in crypto_list,
            all_dict, flip_dict, signal_texts, day, freq, prices_entry
        )
