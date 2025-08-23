import yfinance as yf
import pandas as pd

# If the given stock is without exchange it will be automatically added 
def resolve_ticker(t, start, end):
    candidates = [t, f"{t}.NS", f"{t}.BO"]
    for cand in candidates:
        df = yf.download(cand, start=start, end=end, progress=False, auto_adjust=False, actions=False)
        if not df.empty:
            return cand, df
    raise ValueError(f"Could not download data for {t} (tried: {candidates})")

# For computing RSI
def rsi_wilder(close, window=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

# Computing the given Technical Indicators
def build_dataset(ticker, start, end, use_adj=True):
    tk, df = resolve_ticker(ticker, start, end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df[['Open','High','Low','Close','Adj Close','Volume']].copy()

    # Choose price series for indicators
    price = df['Adj Close'] if use_adj and 'Adj Close' in df.columns else df['Close']

    # Technicals (standard params)
    df['RSI_14'] = rsi_wilder(price, 14)
    df['MA_9']   = price.rolling(9).mean()
    df['MA_21']  = price.rolling(21).mean()
    df['MA_51']  = price.rolling(51).mean()

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    df['MACD']         = ema12 - ema26
    df['MACD_Signal']  = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']    = df['MACD'] - df['MACD_Signal']

    # First fully valid row occurs after MA_51
    df = df.dropna(subset=['RSI_14','MA_9','MA_21','MA_51','MACD','MACD_Signal','MACD_Hist'])
    
    cols = ['Open','High','Low','Close','Adj Close','Volume',
            'RSI_14','MA_9','MA_21','MA_51','MACD','MACD_Signal','MACD_Hist']
    df = df[cols].apply(pd.to_numeric, errors='coerce')
    return tk, df

# ---------- Main Function ----------
if __name__ == "__main__":
    start = "2015-01-01"
    end   = "2025-08-22"
    tk, ds = build_dataset("ONGC.NS", start, end, use_adj=True)
    out = f"Dataset/{tk}_daily_technical.csv"
    # Round off to 6 decimal's 
    ds.to_csv(out, index_label="Date", float_format="%.6f")
    print("Saved:", out)
    print("First valid row date:", ds.index[0].date())
    print(ds.head(3))
