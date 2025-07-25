# Commented out IPython magic to ensure Python compatibility.
import os
import json
import pickle
import pprint
import warnings
import requests
import datetime
import numpy as np
import pandas as pd
import pandas_ta as pta
import plotly.graph_objects as go
import plotly.io as pio

from urllib.request import urlopen
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# Discord webhooks
DISCORD_WEBHOOK_TOKEN  = os.getenv("DISCORD_WEBHOOK_TOKEN")
DISCORD_WEBHOOK_TOKEN2 = os.getenv("DISCORD_WEBHOOK_TOKEN2")
if not DISCORD_WEBHOOK_TOKEN or not DISCORD_WEBHOOK_TOKEN2:
    raise ValueError("Missing DISCORD_WEBHOOK_TOKEN(s)")
DISCORD_WEBHOOK_URL  = f"https://discord.com/api/webhooks/{DISCORD_WEBHOOK_TOKEN}"
DISCORD_WEBHOOK_URL2 = f"https://discord.com/api/webhooks/{DISCORD_WEBHOOK_TOKEN2}"

# Markets → HTML templates
file_paths = {
    "US Market":     "interested_tickers_days_{day}.html",
    "HK Market":     "interested_tickers_hk_days_{day}.html",
    "SNP Market":    "interested_tickers_snp_days_{day}.html",
    "Crypto Market": "interested_tickers_crypto_days_{day}.html",
}

# Markets → image folders
image_folder_paths = {
    "US Market":     "us_images",
    "HK Market":     "hk_images",
    "SNP Market":    None,          # no images for SNP
    "Crypto Market": "crypto_images",
}

# Date range for price history
_today        = datetime.datetime.today()
string_today  = _today.strftime('%Y-%m-%d')
string_1y_ago = (_today - relativedelta(years=1)).strftime('%Y-%m-%d')

# ─── UTILITIES ────────────────────────────────────────────────────────────────

def get_jsonparsed_data(url):
    resp = urlopen(url)
    return json.loads(resp.read().decode())

def prepare_image_dirs():
    for d in image_folder_paths.values():
        if d:
            os.makedirs(d, exist_ok=True)

# ─── PRICE DATA ────────────────────────────────────────────────────────────────

def get_stock_price(symbol, freq='2day'):
    apiKey = os.environ['FMP_API_KEY']
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/"
        f"{symbol}?from={string_1y_ago}&to={string_today}&apikey={apiKey}"
    )
    hist = get_jsonparsed_data(url).get('historical', [])
    df = pd.DataFrame(hist).rename(columns={
        'date':'Date','open':'Open','high':'High',
        'low':'Low','close':'Close','volume':'Volume'
    })
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    if freq == '2day':
        df = df.resample('2D').agg({
            'Open':'first','High':'max','Low':'min',
            'Close':'last','Volume':'sum'
        }).dropna()
    elif freq == 'week':
        df = df.resample('1W').agg({
            'Open':'first','High':'max','Low':'min',
            'Close':'last','Volume':'sum'
        }).dropna()

    df['ATR20']         = pta.atr(df.High, df.Low, df.Close, window=20, mamode='ema')
    df['SMA20']         = df.Close.rolling(20).mean()
    df['SMA50']         = df.Close.rolling(50).mean()
    df['SMA100']        = df.Close.rolling(100).mean()
    df['Ave Volume 20'] = df.Volume.rolling(20).mean()

    return df

# ─── SIGNAL FUNCTIONS ──────────────────────────────────────────────────────────

def is_support_harderv2(df,i):
    cond1 = df.Low[i] < df.Low[i-1]
    cond2 = df.Low[i] < df.Low[i+1]
    cond3 = df.Low[i+1] < df.Low[i+2]
    cond4 = df.Low[i-1] < df.Low[i-2]
    c1 = (df.Low[i-1] - df.Low[i]) > 0.5*df.ATR20[i]
    c2 = (df.Low[i+1] - df.Low[i]) > 0.5*df.ATR20[i]
    c3 = (df.Low[i-1] - df.Low[i]) > 0.8*df.ATR20[i]
    c4 = (df.Low[i+1] - df.Low[i]) > 0.8*df.ATR20[i]
    return (cond1 and cond2 and cond3 and cond4) or (cond1 and cond2 and (c1 and c2) and (c3 or c4))

def is_resistance_harderv2(df,i):
    cond1 = df.High[i] > df.High[i-1]
    cond2 = df.High[i] > df.High[i+1]
    cond3 = df.High[i+1] > df.High[i+2]
    cond4 = df.High[i-1] > df.High[i-2]
    c1 = (df.High[i] - df.High[i-1]) > 0.5*df.ATR20[i]
    c2 = (df.High[i] - df.High[i+1]) > 0.5*df.ATR20[i]
    c3 = (df.High[i] - df.High[i-1]) > 0.8*df.ATR20[i]
    c4 = (df.High[i] - df.High[i+1]) > 0.8*df.ATR20[i]
    return (cond1 and cond2 and cond3 and cond4) or (cond1 and cond2 and (c1 and c2) and (c3 or c4))

def find_levels(df, max_breach):
    lows, highs = [], []
    for i in range(2, len(df)-12):
        if is_support_harderv2(df,i):
            val = df.Low[i]
            if df.Low.iloc[i:max_breach].min() >= val:
                lows.append((df.index[i], val))
        elif is_resistance_harderv2(df,i):
            val = df.High[i]
            if df.High.iloc[i:max_breach].max() <= val:
                highs.append((df.index[i], val))
    return lows, highs

def find_recent_levels(df, start, end):
    lows, highs = [], []
    for k in range(start, end):
        if is_support_harderv2(df,k):
            val = df.Low[k]
            if df.Low.iloc[start:end+1].min() >= val:
                lows.append((k, val))
        elif is_resistance_harderv2(df,k):
            val = df.High[k]
            if df.High.iloc[start:end+1].max() <= val:
                highs.append((k, val))
    return lows, highs

def exe_bull(df,i):
    pin      = ((df.Close[i]-df.Low[i])/(df.High[i]-df.Low[i])>=2/3) and ((df.Open[i]-df.Low[i])/(df.High[i]-df.Low[i])>=2/3)
    markup   = (df.Close[i]>df.Open[i]) and ((df.Close[i]-df.Open[i])/(df.High[i]-df.Low[i])>=2/3)
    icecream = ((df.Close[i]-df.Low[i])/(df.High[i]-df.Low[i])>=2/3) and ((df.Close[i]-df.Open[i])/(df.High[i]-df.Low[i])>=1/2)
    return pin or markup or icecream

def exe_bear(df,i):
    pin      = ((df.Close[i]-df.Low[i])/(df.High[i]-df.Low[i])<=1/3) and ((df.Open[i]-df.Low[i])/(df.High[i]-df.Low[i])<=1/3)
    markup   = (df.Close[i]<df.Open[i]) and ((df.Open[i]-df.Close[i])/(df.High[i]-df.Low[i])>=2/3)
    icecream = ((df.Close[i]-df.Low[i])/(df.High[i]-df.Low[i])<=1/3) and ((df.Open[i]-df.Close[i])/(df.High[i]-df.Low[i])>=1/2)
    return pin or markup or icecream

def bullish_dr1(df,i,drop_days=3):
    exe = exe_bull(df,i)
    cond_price = df.Close[i]>2
    cond_vol = (df['Ave Volume 20'][i]>100000) and (df.Volume[i]>100000)
    j = drop_days-3
    if i==-1:
        s1 = (df.Low[i]<df.Low[i-3:i].min()) and ((df.High[i-3-j:].max()-df.Low[i-3-j:].min())>=5*df.ATR20[i])
        s2 = (df.Low[i-1]<df.Low[i]) and (df.Low[i-1]<df.Low[i-4:i-1].min()) and ((df.High[i-4-j:i].max()-df.Low[i-4-j:i].min())>=5*df.ATR20[i])
        s3 = (df.Low[i-2]<df.Low[i-1:].min()) and (df.Low[i-2]<df.Low[i-5:i-2].min()) and ((df.High[i-5-j:i-1].max()-df.Low[i-5-j:i-1].min())>=5*df.ATR20[i])
        s4 = (df.Low[i-3]<df.Low[i-2:].min()) and (df.Low[i-3]<df.Low[i-6:i-3].min()) and ((df.High[i-6-j:i-2].max()-df.Low[i-6-j:i-2].min())>=5*df.ATR20[i])
    else:
        s1 = (df.Low[i]<df.Low[i-3:i+1].min()) and ((df.High[i-5:i+1].max()-df.Low[i-5:i+1].min())>=4*df.ATR20[i])
        s2 = (df.Low[i-1]<df.Low[i]) and (df.Low[i-1]<df.Low[i-4:i-1].min()) and ((df.High[i-4-j:i-1].max()-df.Low[i-4-j:i-1].min())>=5*df.ATR20[i])
        s3 = (df.Low[i-2]<df.Low[i-1:i+1].min()) and (df.Low[i-2]<df.Low[i-5:i-2].min()) and ((df.High[i-5-j:i-2].max()-df.Low[i-5-j:i-2].min())>=5*df.ATR20[i])
        s4 = (df.Low[i-3]<df.Low[i-2:i+1].min()) and (df.Low[i-3]<df.Low[i-6:i-3].min()) and ((df.High[i-6-j:i-3].max()-df.Low[i-6-j:i-3].min())>=5*df.ATR20[i])
    which = next((idx for idx,(cond) in enumerate((s1,s2,s3,s4),1) if cond), None)
    return (exe and cond_price and cond_vol and any((s1,s2,s3,s4))), which

def bearish_ur1(df,i,rise_days=3):
    exe = exe_bear(df,i)
    cond_price = df.Close[i]>2
    cond_vol = (df['Ave Volume 20'][i]>100000) and (df.Volume[i]>100000)
    j = rise_days-3
    if i==-1:
        s1 = (df.High[i]>df.High[i-3:i].max()) and ((df.High[i-3-j:].max()-df.Low[i-3-j:].min())>=4*df.ATR20[i])
        s2 = (df.High[i-1]>df.High[i]) and (df.High[i-1]>df.High[i-4:i-1].max()) and ((df.High[i-4-j:i-1].max()-df.Low[i-4-j:i-1].min())>=5*df.ATR20[i])
        s3 = (df.High[i-2]>df.High[i-1:].max()) and (df.High[i-2]>df.High[i-5:i-2].max()) and ((df.High[i-5-j:i-2].max()-df.Low[i-5-j:i-2].min())>=5*df.ATR20[i])
        s4 = (df.High[i-3]>df.High[i-2:].max()) and (df.High[i-3]>df.High[i-6:i-3].max()) and ((df.High[i-6-j:i-3].max()-df.Low[i-6-j:i-3].min())>5*df.ATR20[i])
    else:
        s1 = (df.High[i]>df.High[i-4:i+1].max()) and ((df.High[i-6:i+1].max()-df.Low[i-6:i+1].min())>=4*df.ATR20[i])
        s2 = (df.High[i-1]>df.High[i]) and (df.High[i-1]>df.High[i-4:i-1].max()) and ((df.High[i-4-j:i-1].max()-df.Low[i-4-j:i-1].min())>=5*df.ATR20[i])
        s3 = (df.High[i-2]>df.High[i-1:i+1].max()) and (df.High[i-2]>df.High[i-5:i-2].max()) and ((df.High[i-5-j:i-2].max()-df.Low[i-5-j:i-2].min())>=5*df.ATR20[i])
        s4 = (df.High[i-3]>df.High[i-2:i+1].max()) and (df.High[i-3]>df.High[i-6:i-3].max()) and ((df.High[i-6-j:i-3].max()-df.Low[i-6-j:i-3].min())>5*df.ATR20[i])
    which = next((idx for idx,(cond) in enumerate((s1,s2,s3,s4),1) if cond), None)
    return (exe and cond_price and cond_vol and any((s1,s2,s3,s4))), which

def bearish_fs(df,i):
    exe = exe_bear(df,i)
    cond_price = df.Close[i]>2
    cond_vol = (df['Ave Volume 20'][i]>100000) and (df.Volume[i]>100000)
    fs3 = (((df.Low[i-2]<=df.Low[i-1]) and (df.Low[i-2]<=df.Low[i])) and 
           (df.High[i-2]>=df.High[i-1]) and (df.High[i]>df.High[i-2]) and 
           (df.Close[i]<=df.High[i-2]))
    fs4 = (((df.Low[i-3]<=df.Low[i-2]) and (df.Low[i-3]<=df.Low[i-1]) and (df.Low[i-3]<=df.Low[i])) and 
           (df.High[i-3]>=df.High[i-2]) and ((df.High[i]>df.High[i-3]) or (df.High[i-1]>df.High[i-3])) and 
           (df.Close[i]<=df.High[i-3]))
    fs5 = (((df.Low[i-4]<=df.Low[i-3]) and (df.Low[i-4]<=df.Low[i-2]) and (df.Low[i-4]<=df.Low[i-1]) and (df.Low[i-4]<=df.Low[i])) and 
           (df.High[i-4]>=df.High[i-3]) and ((df.High[i]>df.High[i-4]) or (df.High[i-1]>df.High[i-4]) or (df.High[i-2]>df.High[i-4])) and 
           (df.Close[i]<=df.High[i-4]))
    which = next((b for b,cond in zip((3,4,5),(fs3,fs4,fs5)) if cond), None)
    return (exe and cond_price and cond_vol and any((fs3,fs4,fs5))), which

def bullish_fs(df,i):
    exe = exe_bull(df,i)
    cond_price = df.Close[i]>2
    cond_vol = (df['Ave Volume 20'][i]>100000) and (df.Volume[i]>100000)
    fs3 = ((df.High[i-2]>=df.High[i-1]) and (df.High[i-2]>=df.High[i]) and 
           (df.Low[i-2]<=df.Low[i-1]) and (df.Low[i]<df.Low[i-2]) and 
           (df.Close[i]>=df.Low[i-2]))
    fs4 = ((df.High[i-3]>=df.High[i-2]) and (df.High[i-3]>=df.High[i-1]) and (df.High[i-3]>=df.High[i]) and 
           (df.Low[i-3]<=df.Low[i-2]) and ((df.Low[i]<df.Low[i-3]) or (df.Low[i-1]<df.Low[i-3])) and 
           (df.Close[i]>=df.Low[i-3]))
    fs5 = ((df.High[i-4]>=df.High[i-3]) and (df.High[i-4]>=df.High[i-2]) and (df.High[i-4]>=df.High[i-1]) and (df.High[i-4]>=df.High[i]) and 
           (df.Low[i-4]<=df.Low[i-3]) and ((df.Low[i]<df.Low[i-4]) or (df.Low[i-1]<df.Low[i-4]) or (df.Low[i-2]<df.Low[i-4])) and 
           (df.Close[i]>=df.Low[i-4]))
    which = next((b for b,cond in zip((3,4,5),(fs3,fs4,fs5)) if cond), None)
    return (exe and cond_price and cond_vol and any((fs3,fs4,fs5))), which

def test_force_top(df, day, levels):
    for lvl in levels:
        ntl = df.High.iloc[day+1-5] < lvl[1]
        wa  = df.High.iloc[day+1-4:day+1].max() > lvl[1] if day!=-1 else df.High.iloc[day+1-4:].max()>lvl[1]
        cb  = df.Close.iloc[day] <= lvl[1]
        if ntl and wa and cb:
            return True, lvl
    return False, False

def test_force_bottom(df, day, levels):
    for lvl in levels:
        ntl = df.Low.iloc[day+1-5] > lvl[1]
        wb  = df.Low.iloc[day+1-4:day+1].min() < lvl[1] if day!=-1 else df.Low.iloc[day+1-4:].min()<lvl[1]
        ca  = df.Close.iloc[day] >= lvl[1]
        if ntl and wb and ca:
            return True, lvl
    return False, False

def test_sma_above(df,i,j):
    exe = exe_bull(df,i)
    pc  = df.Close[i]>2
    t1  = df.SMA20[i]>df.SMA50[i]
    t2  = df.SMA50[i]>df.SMA100[i]
    bb  = 0.1*df.ATR20[i]
    eb  = 1*df.ATR20[i]
    flow = (df.High[i]>df.SMA50[i]) or (abs(df.High[i]-df.SMA50[i])<bb)
    near = False
    for k in range(i,j,-1):
        near = any([
            (df.High[k]>df.SMA20[k] and df.Low[k]<df.SMA20[k]),
            abs(df.High[k]-df.SMA20[k])<bb,
            abs(df.Low[k]-df.SMA20[k])<bb,
            (df.High[k]>df.SMA50[k] and df.Low[k]<df.SMA50[k]),
            abs(df.High[k]-df.SMA50[k])<bb,
            abs(df.Low[k]-df.SMA50[k])<bb
        ])
        if near: break
    return exe and pc and t1 and t2 and flow and (abs(df.High[i]-df.SMA20[i])<eb or abs(df.High[i]-df.SMA50[i])<eb) and near

def test_sma_below(df,i,j):
    exe = exe_bear(df,i)
    pc  = df.Close[i]>2
    t1  = df.SMA20[i]<df.SMA50[i]
    t2  = df.SMA50[i]<df.SMA100[i]
    bb  = 0.1*df.ATR20[i]
    eb  = 1*df.ATR20[i]
    flow = (df.Low[i]<df.SMA50[i]) or (abs(df.Low[i]-df.SMA50[i])<bb)
    near = False
    for k in range(i,j,-1):
        near = any([
            (df.High[k]>df.SMA20[k] and df.Low[k]<df.SMA20[k]),
            abs(df.High[k]-df.SMA20[k])<bb,
            abs(df.Low[k]-df.SMA20[k])<bb,
            (df.High[k]>df.SMA50[k] and df.Low[k]<df.SMA50[k]),
            abs(df.High[k]-df.SMA50[k])<bb,
            abs(df.Low[k]-df.SMA50[k])<bb
        ])
        if near: break
    return exe and pc and t1 and t2 and flow and (abs(df.Low[i]-df.SMA20[i])<eb or abs(df.Low[i]-df.SMA50[i])<eb) and near

def bullish_uc1(df,i,sma_start=-1,sma_end=-6,rs=-6,re=-26):
    if not test_sma_above(df,sma_start,sma_end): return False,None,None
    lows,highs = find_recent_levels(df,rs,re)
    if not lows or not highs: return False,None,None
    li,lv = lows[0]; hi,hv = highs[0]
    if hi-li>=3 and hv-lv>1.5*df.ATR20[i] and df.Close[i]>=lv:
        for d in range(5):
            if (df.Low[i-d]<lv) and (df.Low.iloc[hi:i-d].min()>lv):
                return True,(df.index[li],lv),(df.index[hi],hv)
    return False,None,None

def bearish_dc1(df,i,sma_start=-1,sma_end=-6,rs=-6,re=-26):
    if not test_sma_below(df,sma_start,sma_end): return False,None,None
    lows,highs = find_recent_levels(df,rs,re)
    if not lows or not highs: return False,None,None
    li,lv = lows[0]; hi,hv = highs[0]
    if hi<li and hv-lv>1.5*df.ATR20[i] and df.Close[i]<=hv:
        for d in range(5):
            if df.High[i-d]>hv and df.High.iloc[li:i-d].max()<hv:
                return True,(df.index[li],lv),(df.index[hi],hv)
    return False,None,None

# ─── SCAN ALL SIGNALS ──────────────────────────────────────────────────────────

def scan_all_signals(tks, day, freq, rd_days, sma_start, sma_end,
                     rs, re, max_br, risk, rr, entry_key):
    all_dict  = {k: [] for k in ['UR1','DR1','bull_fs','bear_fs','bull_fs_sma','bear_fs_sma','UC1','DC1']}
    flip_dict = {}
    for tk in tks:
        try:
            df = get_stock_price(tk, freq=freq)

            bear_ur, ur_sw = bearish_ur1(df, day, rise_days=rd_days)
            bull_dr, dr_sw = bullish_dr1(df, day, drop_days=rd_days)
            bear_fs, bf_sb = bearish_fs(df, day)
            bull_fs, bl_sb = bullish_fs(df, day)
            bull_uc1, ul, uh = bullish_uc1(df, day, sma_start, sma_end, rs, re)
            bear_dc1, dl, dh = bearish_dc1(df, day, sma_start, sma_end, rs, re)

            def add(key, lvl, dirn, swing=None, fsb=None, smakey=None):
                d = {'Ticker':tk,'Levels':lvl,'Direction':dirn}
                if swing is not None: d['Swing Bar']=swing
                if fsb is not None:    d['FS Bar']=fsb
                if smakey is not None: d['SMA Key']=smakey
                d['Prices Entry'] = get_enter_prices(df, day, tk, direction=dirn, risk=risk, ratio=rr)
                all_dict[key].append(d)

            if bear_ur:
                ll,hh = find_levels(df, max_br)
                ok,l = test_force_top(df, day, hh)
                add('UR1', hh, 'Short', swing=ur_sw)
            if bull_dr:
                ll,hh = find_levels(df, max_br)
                ok,l = test_force_bottom(df, day, ll)
                add('DR1', ll, 'Long', swing=dr_sw)
            if bear_fs:
                ll,hh = find_levels(df, max_br)
                add('bear_fs', hh, 'Short', fsb=bf_sb)
                if test_sma_below(df, day, day-5):
                    add('bear_fs_sma', hh, 'Short', fsb=bf_sb, smakey='below')
            if bull_fs:
                ll,hh = find_levels(df, max_br)
                add('bull_fs', ll, 'Long', fsb=bl_sb)
                if test_sma_above(df, day, day-5):
                    add('bull_fs_sma', ll, 'Long', fsb=bl_sb, smakey='above')
            if bull_uc1:
                add('UC1', [ul,uh], 'Long')
            if bear_dc1:
                add('DC1', [dl,dh], 'Short')

        except Exception as e:
            print(f"[{tk}] error:", e)

    for strat, lst in all_dict.items():
        for d in lst:
            flip_dict.setdefault(d['Ticker'], []).append(strat)

    with open('interested_tickers.pickle','wb') as f:
        pickle.dump(all_dict, f)
    with open('flip_dict.pickle','wb') as f:
        pickle.dump(flip_dict, f)

    return all_dict, flip_dict

# ─── PLOTTING & POSTING ───────────────────────────────────────────────────────

def plot_all_with_return(levels, df, day, title, direction, entry, fs_bar=None):
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close
    ))
    # entry->TP
    fig.add_shape(type="rect", x0=df.index[day], x1=df.index[-1],
                  y0=entry['enter'], y1=entry['take_profit'],
                  fillcolor="blue", opacity=0.1)
    # entry->SL
    fig.add_shape(type="rect", x0=df.index[day], x1=df.index[-1],
                  y0=entry['enter'], y1=entry['stop_loss'],
                  fillcolor="red", opacity=0.1)
    for lvl in levels:
        fig.add_shape(type="line", x0=lvl[0], x1=df.index[day],
                      y0=lvl[1], y1=lvl[1], dash="dash")
    if fs_bar:
        fig.add_shape(type="rect",
                      x0=df.index[day+1-fs_bar], x1=df.index[day],
                      y0=df.Low[day+1-fs_bar], y1=df.High[day+1-fs_bar],
                      fillcolor="yellow", opacity=0.2)
    fig.update_layout(title=title, height=600)
    return fig

def process_market(market, tpl, img_folder, filt, all_dict, flip_dict, texts, day, freq, entry_key):
    html_path = tpl.format(day=day)
    now = datetime.datetime.now()
    dt = now.strftime("%Y/%m/%d %H:%M:%S")
    header = f"<h3>Last updated {dt}</h3>"

    lines=[]
    for L in pprint.pformat(flip_dict).splitlines():
        tk=L.split("':")[0].strip(" '")
        if filt(tk):
            lines.append(f"<br>{L}")
            if day==-1:
                texts[market] += "\n"+L

    with open(html_path,'w') as f:
        f.write(header + "".join(lines))
        for strat,lst in all_dict.items():
            f.write(f"<h2>{strat}</h2>")
            for d in lst:
                tk=d['Ticker']
                if not filt(tk): continue
                df=get_stock_price(tk,freq=freq)
                entry=d['Prices Entry'][entry_key]
                d['Volume']=df['Ave Volume 20'][day]
                fig=plot_all_with_return(d['Levels'],df,day,f"{tk}: {strat}",d['Direction'],entry,d.get('FS Bar'))
                info_html=pd.DataFrame.from_dict({k:v for k,v in d.items() if k not in ['Prices Entry','Levels']},orient='index').to_html()
                df_html=pd.DataFrame.from_dict(d['Prices Entry'],orient='index').assign(ticker=tk,date=dt).to_html()
                f.write(info_html+df_html+fig.to_html(full_html=False))
                if img_folder and day==-1:
                    os.makedirs(img_folder,exist_ok=True)
                    png=f"{img_folder}/{tk}_{strat}.png"
                    pio.write_image(fig,png)
                    md=pd.DataFrame.from_dict(d['Prices Entry'],orient='index')[['enter','take_profit','stop_loss','n_shares']].to_markdown(tablefmt='grid')
                    with open(png.replace('.png','.txt'),'w') as tx:
                        tx.write(f"```{md}```")

    if img_folder:
        for fn in os.listdir(img_folder):
            if fn.endswith('.png'):
                txt=open(f"{img_folder}/{fn[:-4]}.txt").read()
                payload={"content":f"{market} signal {fn[:-4]}\n{txt}"}
                with open(f"{img_folder}/{fn}",'rb') as im:
                    requests.post(DISCORD_WEBHOOK_URL2,data=payload,files={"file":(fn,im)})

    for wh in (DISCORD_WEBHOOK_URL,DISCORD_WEBHOOK_URL2):
        with open(html_path,'rb') as h:
            payload={"content":f"{market} signals — download HTML to view\n{texts[market]}", "flags":4096}
            requests.post(wh,data=payload,files={"file":(html_path,h)})

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prepare_image_dirs()

    snp_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    with open("stock_list.txt")  as f: stock_list  = [l.strip() for l in f]
    with open("crypto_list.txt") as f: crypto_list = [l.strip() for l in f]
    universe = list(set(snp_list + stock_list)) + crypto_list

    for day in [-1, -15]:
        freq = '2day'
        rd_days = 5
        sma_s, sma_e = day, day-3
        rs, re = day-30, day-5
        max_br = -6
        risk = 300
        rr = 2
        entry_key = '0.25'

        all_dict, flip_dict = scan_all_signals(
            universe, day, freq, rd_days,
            sma_s, sma_e, rs, re,
            max_br, risk, rr, entry_key
        )

        signal_texts = {m:'' for m in file_paths}

        process_market(
            "US Market", file_paths["US Market"],
            image_folder_paths["US Market"],
            lambda t: not t.endswith('.HK'),
            all_dict, flip_dict, signal_texts,
            day, freq, entry_key
        )
        process_market(
            "HK Market", file_paths["HK Market"],
            image_folder_paths["HK Market"],
            lambda t: t.endswith('.HK'),
            all_dict, flip_dict, signal_texts,
            day, freq, entry_key
        )
        # process_market(
        #     "SNP Market", file_paths["SNP Market"],
        #     image_folder_paths["SNP Market"],
        #     lambda t: t in snp_list,
        #     all_dict, flip_dict, signal_texts,
        #     day, freq, entry_key
        # )
        process_market(
            "Crypto Market", file_paths["Crypto Market"],
            image_folder_paths["Crypto Market"],
            lambda t: t in crypto_list,
            all_dict, flip_dict, signal_texts,
            day, freq, entry_key
        )
