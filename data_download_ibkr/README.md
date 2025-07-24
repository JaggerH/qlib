# IBKR 美股分钟线数据下载脚本

## 1. 安装依赖
```bash
pip install -r requirements.txt
```

## 2. 配置下载参数
编辑 `download_config.yaml`，示例：
```yaml
stocks: [AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA]
start_date: '2023-01-01'
end_date: '2024-05-01'
freq: '1min'           # 支持 '1min', '5min' 等
rth_only: true         # 仅下载RTH（常规交易时段）
timezone: 'US/Eastern' # IBKR数据默认美东时间
```

## 3. 运行下载脚本
```bash
python download_ibkr.py
```

## 4. 数据目录结构
```
raw_data/
  ├── AAPL/
  │   ├── 2024-05-01.csv
  │   ├── 2024-05-02.csv
  │   └── ...
  ├── MSFT/
  │   ├── 2024-05-01.csv
  │   └── ...
  └── ...
```

## 5. CSV文件格式示例
| datetime            | open   | high   | low    | close  | volume | amount  |
|---------------------|--------|--------|--------|--------|--------|---------|
| 2024-05-01 09:30:00 | 170.00 | 170.10 | 169.90 | 170.05 | 1000   | 170050  |
| 2024-05-01 09:31:00 | 170.05 | 170.20 | 170.00 | 170.15 | 800    | 136120  |

## 6. 注意事项
- 需提前配置好IBKR API（TWS或IB Gateway）并保持连接。
- 建议分批下载，避免API限流。
- 下载完成后可用单独脚本转换为Qlib格式。 