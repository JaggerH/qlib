import os
import pandas as pd
from glob import glob

RAW_DATA_DIR = 'raw_data'
QLIB_DATA_DIR = 'qlib_data/1min'
CALENDAR_FILE = 'qlib_data/calendar.minute.txt'
INSTRUMENTS_DIR = 'qlib_data/instruments'
INSTRUMENTS_FILE = os.path.join(INSTRUMENTS_DIR, 'all.txt')
os.makedirs(QLIB_DATA_DIR, exist_ok=True)
os.makedirs(INSTRUMENTS_DIR, exist_ok=True)

symbols = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
all_datetimes = set()
instrument_lines = []

for symbol in symbols:
    all_files = sorted(glob(os.path.join(RAW_DATA_DIR, symbol, '*.csv')))
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file, encoding='utf-8')
            df = df.rename(columns={
                'date': 'datetime',
                'Date': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
            dfs.append(df)
        except Exception as e:
            print(f"处理 {file} 失败: {e}")
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.sort_values('datetime')
        df_all.to_parquet(os.path.join(QLIB_DATA_DIR, f"{symbol}.parquet"), index=False)
        print(f"{symbol} 转换完成，共 {len(df_all)} 行")
        # 收集所有时间戳
        all_datetimes.update(df_all['datetime'].tolist())
        # 记录起止日期
        start_date = df_all['datetime'].iloc[0][:10]
        end_date = df_all['datetime'].iloc[-1][:10]
        instrument_lines.append(f"{symbol} {start_date} {end_date}")
    else:
        print(f"{symbol} 无有效数据")

# 生成 calendar.minute.txt
all_datetimes = sorted(all_datetimes)
with open(CALENDAR_FILE, 'w', encoding='utf-8') as f:
    for dt in all_datetimes:
        f.write(dt + '\n')
print(f"已生成 {CALENDAR_FILE}，共 {len(all_datetimes)} 行")

# 生成 instruments/all.txt
with open(INSTRUMENTS_FILE, 'w', encoding='utf-8') as f:
    for line in instrument_lines:
        f.write(line + '\n')
print(f"已生成 {INSTRUMENTS_FILE}，共 {len(instrument_lines)} 行") 