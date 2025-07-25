# 因子入库流程
和当前因子库进行相关性分析，一般认为 |相关系数| > 0.7 为高度相关，> 0.5 为较强相关。

# 因子注册表工具使用说明

本目录下提供了因子注册表的自动生成、索引、查重、模糊查找和相关性分析工具，方便管理和维护因子。

## 1. 配置 DataHandler

请在本目录下维护 `handler_config.yaml`，指定需要检索的 DataHandler 和相关性分析参数。例如：

```yaml
handlers:
  - module: qlib.contrib.data.loader
    class: Alpha158DL
  - module: factors.custom_factors
    class: CustomFactorDL

corr_params: # 用于计算新因子和已有因子库的相关性
  instruments: ["SH600519"]  # 用于分析的股票代码列表
  start_time: "2022-01-01"  # 分析起始时间
  end_time: "2023-12-31"    # 分析结束时间
```

## 2. 索引（列出）所有因子

输出所有已注册的因子及其公式：

```bash
python factor_inspector.py --list
```

## 3. 检查因子是否已存在

在添加新因子前，建议先检查因子名或公式是否已存在，避免重复：

- 按因子名查重：
  ```bash
  python factor_inspector.py --check --name "MA5"
  ```
- 按公式查重：
  ```bash
  python factor_inspector.py --check --formula "Mean($close, 5)/$close"
  ```
- 同时按名称和公式查重：
  ```bash
  python factor_inspector.py --check --name "MA5" --formula "Mean($close, 5)/$close"
  ```

## 4. 模糊查找因子

支持通过关键字模糊查找因子名或公式，便于检索相关因子：

```bash
python factor_inspector.py --fuzzy "关键字"
```

示例：
```bash
python factor_inspector.py --fuzzy "MA"
python factor_inspector.py --fuzzy "Mean($close"
```

## 5. 相关性分析

可以计算新因子与已有因子库中所有因子的相关性，帮助评估因子的独特性：

```bash
python factor_inspector.py --corr "新因子.csv"
```

注意事项：
- 新因子CSV文件必须包含 `instrument` 和 `datetime` 列作为索引
- 默认因子值列名为 `factor`
- 相关性分析会输出前10个相关性最高的因子
- 相关性绝对值 > 0.5 的因子会被标记为【重点】，需要特别关注
- 首次运行会在 `cache` 目录下生成 `handler.csv` 缓存文件
- 如果因子数量发生变化，会自动重新生成缓存

## 6. 规范建议

- **每次添加新因子前，务必先用查重功能检查，避免名称或公式重复**
- **如需批量检索相关因子，优先使用模糊查找功能**
- **新因子入库前，建议进行相关性分析，避免加入高度相关的冗余因子**
- **如需扩展 DataHandler，请同步维护 `handler_config.yaml`**
- **相关性分析参数（股票、时间范围等）请根据实际需求在配置文件中调整**

---

如有更多需求或建议，请补充本说明文档。
