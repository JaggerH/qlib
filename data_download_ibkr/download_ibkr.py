import os
import yaml
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from ib_insync import IB, Stock, util

RAW_DATA_DIR = 'raw_data'

# 读取配置
with open('download_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

stocks = config['stocks']
start_date = datetime.strptime(config['start_date'], '%Y-%m-%d')
end_date = datetime.strptime(config['end_date'], '%Y-%m-%d')
freq = config.get('freq', '1min')
rth_only = config.get('rth_only', True)
timezone = config.get('timezone', 'US/Eastern')

# IBKR连接（模拟端口7497，真实账户4001/4002）
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # 模拟端口

def daterange(start, end):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

for symbol in stocks:
    symbol_dir = os.path.join(RAW_DATA_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    for single_date in tqdm(list(daterange(start_date, end_date)), desc=f"{symbol}"):
        date_str = single_date.strftime('%Y-%m-%d')
        out_path = os.path.join(symbol_dir, f"{date_str}.csv")
        if os.path.exists(out_path):
            continue  # 已下载跳过

        # IBKR历史数据请求
        contract = Stock(symbol, 'SMART', 'USD')
        # IBKR历史数据API要求endDateTime为美东时间字符串
        end_dt = single_date + timedelta(days=1)
        end_dt_str = end_dt.strftime('%Y%m%d 00:00:00 US/Eastern')
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_dt_str,
                durationStr='1 D',
                barSizeSetting=freq,
                whatToShow='TRADES',
                useRTH=rth_only,
                formatDate=1,
                keepUpToDate=False
            )
            df = util.df(bars)
            if df.empty:
                continue
            # 字段标准化
            df = df.rename(columns={
                'date': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'average': 'amount'  # IBKR没有amount字段，这里用average或留空
            })
            # 只保留需要的字段
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df.to_csv(out_path, index=False)
        except Exception as e:
            print(f"下载 {symbol} {date_str} 失败: {e}")

ib.disconnect()
print("数据下载完成。请检查 raw_data/ 目录。") 