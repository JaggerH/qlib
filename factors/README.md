# 因子注册表工具使用说明

本目录下提供了因子注册表的自动生成、索引、查重和模糊查找工具，方便管理和维护因子。

## 1. 配置 DataHandler

请在本目录下维护 `handler_config.yaml`，指定需要检索的 DataHandler。例如：

```yaml
handlers:
  - module: qlib.contrib.data.loader
    class: Alpha158DL
  - module: factors.custom_factors
    class: CustomFactorDL
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

## 5. 规范建议

- **每次添加新因子前，务必先用查重功能检查，避免名称或公式重复。**
- **如需批量检索相关因子，优先使用模糊查找功能。**
- **如需扩展 DataHandler，请同步维护 `factor_handler_config.yaml`。**

---

如有更多需求或建议，请补充本说明文档。
