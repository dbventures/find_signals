# Commented out IPython magic to ensure Python compatibility.
import pandas_ta as pta
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import pickle
import os
import requests
import json
import datetime
from urllib.request import urlopen
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings("ignore")

DISCORD_WEBHOOK_TOKEN = os.getenv("DISCORD_WEBHOOK_TOKEN")
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

# To reference specific dates when scraping for stock prices
today = datetime.datetime.today()
string_today   = today.strftime('%Y-%m-%d')
string_1y_ago  = (today - relativedelta(years=1)).strftime('%Y-%m-%d')
string_4y_ago  = (today - relativedelta(years=4)).strftime('%Y-%m-%d')
string_5d_ago  = (today - relativedelta(days=5)).strftime('%Y-%m-%d')

# send to discord later
file_paths = {
    "US Market":    f"interested_tickers_days_{{day}}.html",
    "HK Market":    f"interested_tickers_hk_days_{{day}}.html",
    "Crypto Market":f"interested_tickers_crypto_days_{{day}}.html",
    "SNP Market":   f"interested_tickers_snp_days_{{day}}.html",
}
image_folder_paths = {
    "US Market":    "us_images",
    "HK Market":    "hk_images",
    "Crypto Market":"crypto_images",
}

# Ensure image folders exist
def prepare_image_dirs():
    for folder in image_folder_paths.values():
        os.makedirs(folder, exist_ok=True)

# get_stock_price using FMP instead of yahooquery
# get stock prices using yfinance library
def get_stock_price(symbol, freq='2day'):
    apiKey = os.environ['FMP_API_KEY']
    # Fetch daily historical from FMP
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/"
           f"{symbol}?from={string_1y_ago}&to={string_today}&apikey={apiKey}")
    data = get_jsonparsed_data(url).get('historical', [])
    df = pd.DataFrame(data)
    # rename to match yfinance output
    df = df.rename(columns={
        'date':   'Date',
        'open':   'Open',
        'high':   'High',
        'low':    'Low',
        'close':  'Close',
        'volume': 'Volume'
    })
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    # now resample if needed
    if freq == '2day':
        df = df.resample('2D').agg({
            'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'
        }).dropna(how='any')
    elif freq == 'week':
        df = df.resample('1W').agg({
            'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'
        }).dropna(how='any')
    # compute indicators exactly as before
    df['ATR20']         = pta.atr(df['High'], df['Low'], df['Close'], window=20, fillna=False, mamode='ema')
    df['SMA20']         = df['Close'].rolling(20).mean()
    df['SMA50']         = df['Close'].rolling(50).mean()
    df['SMA100']        = df['Close'].rolling(100).mean()
    df['Ave Volume 20'] = df['Volume'].rolling(20).mean()
    return df

# (All your fractal/signal functions unchanged: is_support, is_resistance, ..., bullish_uc1, bearish_dc1)

# ... [signal‐finding definitions go here, exactly as before] ...

# Perform your scan, filling all_dict and flip_dict (unchanged logic)
def scan_all_signals(stock_list_all, day, freq, rise_drop_days, risk, risk_reward_ratio, 
                     sma_start, sma_end, recent_swing_start, recent_swing_end, max_breach, 
                     prices_entry):
    all_dict = {'UR1':[],'DR1':[],
                'bull_fs':[],'bear_fs':[], 'bull_fs_sma':[], 'bear_fs_sma':[],
                'UC1': [], 'DC1': []}
    flip_dict = {}
    for i, ticker in enumerate(stock_list_all):
        try:
            df = get_stock_price(ticker, freq=freq)
            # your existing signal calls:
            bearish_ur_result, swing_bar_ur = bearish_ur1(df,day,rise_days=rise_drop_days)
            bullish_dr_result, swing_bar_dr = bullish_dr1(df,day,drop_days=rise_drop_days)
            bearish_fs_result, which_bar_bear_fs = bearish_fs(df,day)
            bullish_fs_result, which_bar_bull_fs = bullish_fs(df,day)
            bullish_uc1_result, most_recent_low, most_recent_high = bullish_uc1(
                df, day, sma_start=sma_start, sma_end=sma_end,
                recent_swing_start=recent_swing_start, recent_swing_end=recent_swing_end
            )
            # … (rest of your per‐ticker logic exactly as before) …

        except Exception as e:
            print(f'({i}) Error for {ticker}: {e}')

    # serialize as before
    with open('interested_tickers.pickle', 'wb') as handle:
        pickle.dump(all_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('flip_dict.pickle', 'wb') as handle:
        pickle.dump(flip_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_dict, flip_dict

# Common HTML + Discord sending for any market
def process_market(market, file_template, image_folder, filter_fn,
                   all_dict, flip_dict, signal_texts, day):
    file_path = file_template.format(day=day)
    # build HTML
    now       = datetime.datetime.now()
    dt_string = now.strftime("%m/%d/%Y %I:%M:%S %p")
    tz        = now.astimezone().tzname()
    header    = (
        "<h3>Last updated: <span id='timestring'></span></h3>"
        f"<script>var date = new Date('{dt_string} {tz}');"
        "document.getElementById('timestring').innerHTML += date.toString()</script>"
    )

    html_lines = []
    for line in pd.pprint.pformat(flip_dict).splitlines():
        ticker = line.split("':")[0].strip(" '")
        if filter_fn(ticker):
            html_lines.append(f'<br/>{line}')
            if day == -1:
                signal_texts[market] += '\n' + line
    html_body = header + ''.join(html_lines)

    with open(file_path, 'w') as f:
        f.write(html_body)
        for strat, lst in all_dict.items():
            f.write(f"<h2>{strat}</h2>")
            for td in lst:
                t = td['Ticker']
                if not filter_fn(t): 
                    continue
                # reuse your existing block that builds the DataFrame, Figure, etc.
                df      = get_stock_price(t, freq=freq)
                entry   = td['Prices Entry'][prices_entry]
                td['Volume'] = df['Ave Volume 20'][day]
                # plot
                fig = plot_all_with_return(
                    td['Levels'], df, day,
                    f"{t}: {strat}", td['Direction'],
                    entry, fs_bar=td.get('FS Bar', None)
                )
                # metadata table
                mdict = td.copy()
                pdict = mdict.pop('Prices Entry')
                mdict.pop('Levels')
                info_html = pd.DataFrame.from_dict(mdict, orient='index').to_html()
                df_html   = (pd.DataFrame.from_dict(pdict, orient='index')
                             .assign(direction=td['Direction'],
                                     volume=td['Volume'],
                                     ticker=t,
                                     date=dt_string,
                                     value=lambda d: d['n_shares'] * d['enter'],
                                     strategy=strat)
                             .reset_index()
                             [['date','ticker','direction','volume','index','enter',
                               'take_profit','stop_loss','n_shares','more_than_atr',
                               'value','strategy']]
                             .to_html()
                )
                f.write(info_html + df_html + fig.to_html(full_html=False, include_plotlyjs='cdn'))
                # save images/text for Discord if US/HK/Crypto
                if image_folder and day == -1:
                    pio.write_image(fig, f"{image_folder}/{t}_{strat}.png", width=1400, height=800)
                    markdown = (
                        pd.DataFrame.from_dict(pdict, orient='index')
                          [['enter','take_profit','stop_loss','n_shares']]
                          .to_markdown(tablefmt='grid')
                    )
                    with open(f"{image_folder}/{t}_{strat}.txt","w") as txt:
                        txt.write(f"```{markdown}```")

    # Discord: send images first
    if image_folder:
        for img in os.listdir(image_folder):
            if img.lower().endswith(('.png','.jpg','.jpeg')):
                txt = open(f"{image_folder}/{img[:-4]}.txt").read()
                payload = {
                    "content": (
                        f"Signal detected for the {market}.\n"
                        "For interactive charts, please DOWNLOAD the HTML file at the end! :)\n\n"
                        f"{img[:-4]}\n{txt}"
                    )
                }
                with open(f"{image_folder}/{img}","rb") as im:
                    files = {"file": (img, im, "image/png")}
                    requests.post(DISCORD_WEBHOOK_URL2, data=payload, files=files)

    # then send the HTML page
    for webhook in (DISCORD_WEBHOOK_URL, DISCORD_WEBHOOK_URL2):
        with open(file_path,"rb") as f:
            files = {"file": (file_path, f, "text/html")}
            payload = {
                "content": (
                    f"These are the current signals for the {market}.\n"
                    "Please DOWNLOAD the HTML file and open in your browser to view! :)\n"
                    f"{signal_texts[market]}"
                ),
                "flags": 4096
            }
            requests.post(webhook, data=payload, files=files)


# ===  main ===
prepare_image_dirs()

for day in [-1, -15]:
    freq                = '2day'
    rise_drop_days      = 5
    atr_multiple_urdr   = 5
    sma_start           = day
    sma_end             = day-3
    recent_swing_start  = day-30
    recent_swing_end    = day-5
    risk                = 300
    max_breach          = -6
    prices_entry        = '0.25'
    risk_reward_ratio   = 2

    # build your full stock_list_all (unchanged)
    # … (loading stock_list_snp, crypto_list, stock_list, etc.) …
    stock_list_all = list(set(stock_list_snp).union(stock_list)) + crypto_list

    # scan and get results
    all_dict, flip_dict = scan_all_signals(
        stock_list_all, day, freq, rise_drop_days, risk,
        risk_reward_ratio, sma_start, sma_end,
        recent_swing_start, recent_swing_end, max_breach,
        prices_entry
    )

    # prepare per‐market text buffers
    signal_texts = {m: '' for m in file_paths}

    # process each market
    process_market(
        "US Market",
        file_paths["US Market"],
        image_folder_paths["US Market"],
        lambda t: not t.endswith('.HK'),
        all_dict, flip_dict, signal_texts, day
    )
    process_market(
        "HK Market",
        file_paths["HK Market"],
        image_folder_paths["HK Market"],
        lambda t: t.endswith('.HK'),
        all_dict, flip_dict, signal_texts, day
    )
    process_market(
        "SNP Market",
        file_paths["SNP Market"],
        None,  # no images for SNP
        lambda t: t in stock_list_snp,
        all_dict, flip_dict, signal_texts, day
    )
    process_market(
        "Crypto Market",
        file_paths["Crypto Market"],
        image_folder_paths["Crypto Market"],
        lambda t: t in crypto_list,
        all_dict, flip_dict, signal_texts, day
    )
