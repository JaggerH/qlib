#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CustomHandler - 整合qlib expression和Python code的数据处理器

这个模块提供了一个自定义DataHandler，能够：
1. 支持qlib表达式语法
2. 支持TA-Lib技术指标计算
3. 支持自定义Python代码
4. 将多种数据源整合到一个统一的接口中

使用说明：
----------

1. 基本用法 - 直接传入factors列表：
   ```python
   factors = [
       {
           "name": "RSI14",
           "talib_function": "RSI",
           "params": {"input_col": "close", "timeperiod": 14},
           "source": "TA-Lib",
           "description": "14日相对强弱指标"
       },
       {
           "name": "ma20",
           "qlib_expression": "Mean($close, 20)",
           "source": "custom_factor",
           "description": "20日移动平均"
       }
   ]

   handler = CustomHandler(
       instruments=["SH600000", "SH600036"],
       start_time="2023-01-01",
       end_time="2023-12-31",
       factors=factors
   )
   ```

2. 从YAML文件加载：
   ```python
   handler = create_handler_from_yaml(
       "factors.yaml",
       instruments=["SH600000", "SH600036"],
       start_time="2023-01-01",
       end_time="2023-12-31"
   )
   ```

因子配置格式：
-------------

每个因子必须包含 "name" 字段，然后根据类型包含以下字段之一：

1. qlib表达式因子：
   - qlib_expression: qlib表达式字符串

2. TA-Lib技术指标因子：
   - talib_function: TA-Lib函数名
   - inputs: 输入列列表（可选，用于多输入函数如ADX）
   - params/talib_params: 参数字典

3. 自定义Python函数因子：
   - python_function: 可调用的Python函数

支持的TA-Lib函数：
-----------------
- 单输入函数：RSI, SMA, EMA, WMA, DEMA, TEMA, TRIMA等
- 多输入函数：ADX, STOCH, MACD, BBANDS, AROON等
- 成交量函数：OBV, AD, ADOSC等

注意事项：
---------
1. 所有TA-Lib函数都会自动处理输入数据类型转换
2. 多输出函数（如MACD, STOCH）会自动生成多个列
3. 列名会自动处理$前缀问题
4. Python函数接收清理后的DataFrame（去掉$前缀）
"""

import pandas as pd
import numpy as np
import warnings
import yaml
import traceback

# qlib imports
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader

# Optional TA-Lib import
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


class CustomHandler(DataHandlerLP):
    """
    自定义数据处理器，支持：
    1. qlib表达式 (通过QlibDataLoader)
    2. TA-Lib技术指标
    3. 自定义Python代码计算

    输入格式遵循factors.yaml标准格式
    """

    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        factors=None,
        **kwargs,
    ):
        """
        初始化CustomHandler

        Parameters:
        -----------
        instruments : str or list
            股票池
        start_time/end_time : str
            时间范围
        freq : str
            数据频率
        factors : list
            因子配置列表，遵循factors.yaml格式
            每个因子可包含:
            - name: 因子名称
            - qlib_expression: qlib表达式 (qlib类型因子)
            - talib_function: TA-Lib函数名 (TA-Lib类型因子)
            - inputs: 输入列列表 (TA-Lib函数)
            - params/talib_params: 参数字典
            - python_function: Python函数 (自定义函数类型)
        """

        # 如果没有提供factors，尝试从factors.yaml文件加载
        if factors is None:
            factors = self._load_factors_from_yaml()

        # 解析因子配置
        self.factors = factors or []
        self.qlib_expressions = {}
        self.talib_configs = {}
        self.python_funcs = {}
        self.python_codes = {}  # 新增：支持python_code字段

        self._parse_factors()

        # 构造基础数据加载器 (qlib原生数据)
        base_config = self._get_base_config()
        qlib_config = self._get_qlib_expressions_config()

        # 合并配置
        if qlib_config:
            all_config = {"base": base_config, "expressions": qlib_config}
        else:
            all_config = base_config

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": all_config,
                "swap_level": False,
                "freq": freq,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            **kwargs,
        )

    def _load_factors_from_yaml(self):
        """尝试从factors.yaml文件加载因子配置"""
        import os

        yaml_path = "factors/factors.yaml"

        # 尝试多个可能的路径
        possible_paths = [
            yaml_path,
            os.path.join(os.path.dirname(__file__), "factors.yaml"),
            "factors.yaml",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    factors = load_factors_from_yaml(path)
                    print(f"从 {path} 加载了 {len(factors)} 个因子")
                    return factors
                except Exception as e:
                    print(f"从 {path} 加载因子失败: {e}")
                    continue

        print("未找到factors.yaml文件，使用空因子列表")
        return []

    def _parse_factors(self):
        """解析factors.yaml格式的因子配置"""
        for factor in self.factors:
            name = factor.get("name")
            if not name:
                continue

            # qlib表达式类型
            if "qlib_expression" in factor:
                self.qlib_expressions[name] = factor["qlib_expression"]

            # TA-Lib技术指标类型
            elif "talib_function" in factor:
                talib_config = {
                    "func": factor["talib_function"],
                    "inputs": factor.get("inputs", []),
                    "params": {},
                }

                # 支持两种参数格式
                if "params" in factor:
                    talib_config["params"].update(factor["params"])
                if "talib_params" in factor:
                    talib_config["params"].update(factor["talib_params"])

                self.talib_configs[name] = talib_config

            # 自定义Python函数类型
            elif "python_function" in factor:
                self.python_funcs[name] = factor["python_function"]

            # 自定义Python代码类型
            elif "python_code" in factor:
                self.python_codes[name] = {
                    "code": factor["python_code"],
                    "inputs": factor.get("inputs", []),
                    "outputs": factor.get("outputs", ["result"]),
                }

    def _get_base_config(self):
        """获取基础数据配置(OHLCV等)"""
        fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
        names = ["open", "high", "low", "close", "volume", "vwap"]
        return fields, names

    def _get_qlib_expressions_config(self):
        """构造qlib表达式配置"""
        if not self.qlib_expressions:
            return None

        fields = []
        names = []

        for name, expr in self.qlib_expressions.items():
            fields.append(expr)
            names.append(name)

        return fields, names

    def fetch(
        self,
        selector=slice(None, None),
        level="datetime",
        col_set="__all",
        data_key="infer",
    ):
        """
        重写fetch方法，添加TA-Lib和自定义Python计算
        """
        # 首先获取基础数据
        df = super().fetch(selector, level, col_set, data_key)

        if df.empty:
            return df

        # 添加TA-Lib指标
        df = self._add_talib_indicators(df)

        # 添加自定义Python计算
        df = self._add_python_calculations(df)

        # 添加自定义Python代码计算
        df = self._add_python_code_calculations(df)

        # 过滤掉基础数据列，只保留计算出的因子
        factor_columns = self._get_factor_columns()
        if factor_columns:
            # 只保留配置的因子列
            available_factor_cols = [col for col in factor_columns if col in df.columns]
            if available_factor_cols:
                df = df[available_factor_cols]
            else:
                # 如果没有找到任何因子列，返回空DataFrame但保持索引结构
                df = df.iloc[:, 0:0]
        else:
            # 如果没有配置任何因子，返回空DataFrame但保持索引结构
            df = df.iloc[:, 0:0]

        return df

    def _get_factor_columns(self):
        """获取应该保留的因子列名"""
        factor_columns = []

        # qlib表达式因子
        factor_columns.extend(self.qlib_expressions.keys())

        # TA-Lib因子
        factor_columns.extend(self.talib_configs.keys())

        # Python函数因子
        factor_columns.extend(self.python_funcs.keys())

        # Python代码因子
        factor_columns.extend(self.python_codes.keys())

        return factor_columns

    def _add_talib_indicators(self, df):
        """添加TA-Lib技术指标"""
        if not TALIB_AVAILABLE or not self.talib_configs:
            return df

        # 按股票分组计算TA-Lib指标
        def apply_talib_to_group(group_df):
            """对单个股票应用TA-Lib函数 - 通用实现支持所有TA-Lib函数"""
            result_df = group_df.copy()

            for indicator_name, config in self.talib_configs.items():
                try:
                    func_name = config["func"]
                    inputs = config.get("inputs", [])
                    params = config.get("params", {}).copy()  # 使用副本避免修改原配置

                    # 获取TA-Lib函数
                    if not hasattr(talib, func_name):
                        warnings.warn(f"TA-Lib function {func_name} not found")
                        continue

                    talib_func = getattr(talib, func_name)

                    # 动态处理输入数据
                    input_arrays = []

                    if inputs:
                        # 如果明确指定了inputs
                        for input_col in inputs:
                            actual_col = self._resolve_column_name(input_col, group_df)
                            if actual_col is not None:
                                values = self._prepare_talib_input(group_df[actual_col])
                                if values is not None:
                                    input_arrays.append(values)
                                else:
                                    warnings.warn(
                                        f"Invalid data in column {actual_col} for {indicator_name}"
                                    )
                                    break
                            else:
                                warnings.warn(
                                    f"Column {input_col} not found for {indicator_name}. Available: {list(group_df.columns)}"
                                )
                                break
                    else:
                        # 自动推断输入（适用于单输入函数）
                        input_col = params.get("input_col", "close")
                        actual_col = self._resolve_column_name(input_col, group_df)
                        if actual_col is not None:
                            values = self._prepare_talib_input(group_df[actual_col])
                            if values is not None:
                                input_arrays.append(values)
                                # 从参数中移除input_col
                                params.pop("input_col", None)
                            else:
                                warnings.warn(
                                    f"Invalid data in column {actual_col} for {indicator_name}"
                                )
                                continue
                        else:
                            warnings.warn(
                                f"Column {input_col} not found for {indicator_name}. Available: {list(group_df.columns)}"
                            )
                            continue

                    # 如果输入数据准备完成，调用TA-Lib函数
                    expected_inputs = len(inputs) if inputs else 1
                    if input_arrays and len(input_arrays) == expected_inputs:
                        # 验证数据长度
                        min_length = min(len(arr) for arr in input_arrays)
                        if min_length < 2:
                            warnings.warn(
                                f"Insufficient data length ({min_length}) for {indicator_name}"
                            )
                            continue

                        # 调用TA-Lib函数
                        result = talib_func(*input_arrays, **params)

                        # 处理返回结果
                        if isinstance(result, tuple):
                            # 多输出函数（如MACD, STOCH等）
                            output_names = self._get_output_names(
                                func_name, indicator_name
                            )
                            for i, output in enumerate(result):
                                if i < len(output_names):
                                    result_df[output_names[i]] = output
                        else:
                            # 单输出函数
                            result_df[indicator_name] = result

                        # 调试信息
                        if isinstance(result, tuple):
                            if len(result) > 0:
                                valid_count = np.sum(~np.isnan(result[0]))
                                total_count = len(result[0])
                                print(
                                    f"TA-Lib {indicator_name} calculated: {valid_count} valid values from {total_count} total (tuple output)"
                                )
                        elif hasattr(result, "__len__"):
                            valid_count = np.sum(~np.isnan(result))
                            total_count = len(result)
                            print(
                                f"TA-Lib {indicator_name} calculated: {valid_count} valid values from {total_count} total"
                            )

                except Exception as e:
                    warnings.warn(
                        f"Failed to calculate TA-Lib indicator {indicator_name}: {e}\n{traceback.format_exc()}"
                    )

            return result_df

        # 按instrument分组应用TA-Lib
        if "instrument" in df.index.names:
            df = df.groupby(level="instrument", group_keys=False).apply(
                apply_talib_to_group
            )
        else:
            df = apply_talib_to_group(df)

        return df

    def _prepare_talib_input(self, series):
        """
        准备TA-Lib函数的输入数据

        Parameters:
        -----------
        series : pd.Series
            输入数据系列

        Returns:
        --------
        np.array or None : 处理后的数据数组，如果数据无效则返回None
        """
        try:
            # 转换为float64
            values = series.astype(np.float64)

            # 检查数据质量
            if len(values) == 0:
                return None

            # 检查有效数据比例
            valid_count = np.sum(~np.isnan(values))
            if valid_count < len(values) * 0.5:  # 至少50%有效数据
                warnings.warn(
                    f"Too many NaN values in input data: {valid_count}/{len(values)} valid"
                )
                return None

            return values

        except Exception as e:
            warnings.warn(f"Failed to prepare TA-Lib input data: {e}")
            return None

    def _resolve_column_name(self, col_name, df):
        """解析列名，处理$前缀"""
        # 尝试原始列名
        if col_name in df.columns:
            return col_name
        # 尝试添加$前缀
        elif f"${col_name}" in df.columns:
            return f"${col_name}"
        # 尝试去掉$前缀
        elif col_name.startswith("$") and col_name[1:] in df.columns:
            return col_name[1:]
        else:
            # 调试信息
            print(
                f"Column '{col_name}' not found. Available columns: {list(df.columns)}"
            )
            return None

    def _get_output_names(self, func_name, base_name):
        """获取多输出TA-Lib函数的输出名称"""
        # 定义常见多输出函数的输出名称
        output_mappings = {
            "MACD": ["MACD", "MACD_signal", "MACD_hist"],
            "STOCH": ["slowk", "slowd"],
            "STOCHF": ["fastk", "fastd"],
            "STOCHRSI": ["fastk", "fastd"],
            "BBANDS": ["upperband", "middleband", "lowerband"],
            "AROON": ["aroondown", "aroonup"],
            "AROONOSC": ["aroonosc"],
            "PLUS_DI": ["plus_di"],
            "MINUS_DI": ["minus_di"],
            "PLUS_DM": ["plus_dm"],
            "MINUS_DM": ["minus_dm"],
        }

        if func_name in output_mappings:
            return [f"{base_name}_{suffix}" for suffix in output_mappings[func_name]]
        else:
            # 默认情况，使用序号
            return [f"{base_name}_output_{i}" for i in range(10)]  # 假设最多10个输出

    def _add_python_calculations(self, df):
        """添加自定义Python计算"""
        if not self.python_funcs:
            return df

        def apply_python_to_group(group_df):
            """对单个股票应用Python函数"""
            result_df = group_df.copy()

            # 创建列名映射，去掉$前缀方便Python函数使用
            clean_df = group_df.copy()
            for col in group_df.columns:
                if col.startswith("$"):
                    clean_name = col[1:]  # 去掉$前缀
                    clean_df = clean_df.rename(columns={col: clean_name})

            for func_name, func in self.python_funcs.items():
                try:
                    if callable(func):
                        # 使用清理后的列名调用函数
                        result = func(clean_df)
                        result_df[func_name] = result
                    else:
                        warnings.warn(f"Python function {func_name} is not callable")
                except Exception as e:
                    warnings.warn(f"Failed to apply Python function {func_name}: {e}")

            return result_df

        # 按instrument分组应用Python函数
        if "instrument" in df.index.names:
            df = df.groupby(level="instrument", group_keys=False).apply(
                apply_python_to_group
            )
        else:
            df = apply_python_to_group(df)

        return df

    def _add_python_code_calculations(self, df):
        """添加自定义Python代码计算"""
        if not self.python_codes:
            return df

        def apply_python_code_to_group(group_df):
            """对单个股票应用Python代码"""
            result_df = group_df.copy()

            # 创建列名映射，去掉$前缀方便Python代码使用
            clean_df = group_df.copy()
            for col in group_df.columns:
                if col.startswith("$"):
                    clean_name = col[1:]  # 去掉$前缀
                    clean_df = clean_df.rename(columns={col: clean_name})

            for code_name, config in self.python_codes.items():
                try:
                    code = config["code"]
                    inputs = config.get("inputs", [])
                    outputs = config.get("outputs", ["result"])

                    # 准备输入变量
                    local_vars = {}

                    # 添加基础数据
                    for input_name in inputs:
                        if input_name in clean_df.columns:
                            # 保持pandas Series格式，但确保数据类型为float64
                            local_vars[input_name] = clean_df[input_name].astype(
                                np.float64
                            )
                        else:
                            warnings.warn(
                                f"Input column {input_name} not found for {code_name}"
                            )
                            continue

                    # 添加data变量（用于索引）
                    local_vars["data"] = clean_df

                    # 执行Python代码
                    exec(code, globals(), local_vars)

                    # 获取输出结果
                    for output_name in outputs:
                        if output_name in local_vars:
                            result = local_vars[output_name]
                            if hasattr(result, "index"):
                                # 如果是pandas Series，直接使用
                                result_df[code_name] = result
                            else:
                                # 如果是numpy array，创建Series
                                result_df[code_name] = pd.Series(
                                    result, index=clean_df.index
                                )
                        else:
                            warnings.warn(
                                f"Output {output_name} not found in {code_name}"
                            )

                except Exception as e:
                    warnings.warn(f"Failed to execute Python code {code_name}: {e}")
                    import traceback

                    print(f"Error details: {traceback.format_exc()}")

            return result_df

        # 按instrument分组应用Python代码
        if "instrument" in df.index.names:
            df = df.groupby(level="instrument", group_keys=False).apply(
                apply_python_code_to_group
            )
        else:
            df = apply_python_code_to_group(df)

        return df


def create_sample_handler():
    """创建示例CustomHandler - 使用factors.yaml格式"""

    # 示例Python函数
    def price_momentum(df):
        """价格动量指标"""
        return (df["close"] - df["close"].shift(5)) / df["close"].shift(5)

    def volume_ratio(df):
        """成交量比率"""
        return df["volume"] / df["volume"].rolling(20).mean()

    def bollinger_width(df):
        """布林带宽度"""
        ma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        return (ma20 + 2 * std20) - (ma20 - 2 * std20)

    # 使用factors.yaml标准格式
    factors = [
        # qlib表达式因子
        {
            "name": "return_1d",
            "qlib_expression": "Ref($close, -1) / $close - 1",
            "source": "custom_factor",
            "description": "1日收益率",
            "tags": ["收益率", "动量类"],
        },
        {
            "name": "ma5",
            "qlib_expression": "Mean($close, 5)",
            "source": "custom_factor",
            "description": "5日移动平均",
            "tags": ["技术指标", "趋势类"],
        },
        {
            "name": "ma20",
            "qlib_expression": "Mean($close, 20)",
            "source": "custom_factor",
            "description": "20日移动平均",
            "tags": ["技术指标", "趋势类"],
        },
        {
            "name": "volatility",
            "qlib_expression": "Std($close, 20)",
            "source": "custom_factor",
            "description": "20日波动率",
            "tags": ["波动率"],
        },
        {
            "name": "high_low_ratio",
            "qlib_expression": "$high / $low - 1",
            "source": "custom_factor",
            "description": "高低价比率",
            "tags": ["量价类", "日内变动"],
        },
        # TA-Lib技术指标因子
        {
            "name": "RSI14",
            "talib_function": "RSI",
            "params": {"input_col": "close", "timeperiod": 14},
            "source": "TA-Lib",
            "description": "14日相对强弱指标",
            "tags": ["技术指标", "动量类", "震荡指标"],
        },
        {
            "name": "SMA10",
            "talib_function": "SMA",
            "params": {"input_col": "close", "timeperiod": 10},
            "source": "TA-Lib",
            "description": "10日简单移动平均",
            "tags": ["技术指标", "趋势类", "移动平均"],
        },
        {
            "name": "EMA12",
            "talib_function": "EMA",
            "params": {"input_col": "close", "timeperiod": 12},
            "source": "TA-Lib",
            "description": "12日指数移动平均",
            "tags": ["技术指标", "趋势类", "移动平均"],
        },
        {
            "name": "ADX14",
            "talib_function": "ADX",
            "inputs": ["$high", "$low", "$close"],
            "params": {"timeperiod": 14},
            "source": "TA-Lib",
            "description": "14日平均趋向指标",
            "tags": ["技术指标", "趋势强度"],
        },
        {
            "name": "MACD",
            "talib_function": "MACD",
            "params": {
                "input_col": "close",
                "fastperiod": 12,
                "slowperiod": 26,
                "signalperiod": 9,
            },
            "source": "TA-Lib",
            "description": "MACD指标",
            "tags": ["技术指标", "趋势类", "动量类"],
        },
        # 自定义Python函数因子
        {
            "name": "momentum_5d",
            "python_function": price_momentum,
            "source": "custom_factor",
            "description": "5日价格动量",
            "tags": ["动量类", "自定义"],
        },
        {
            "name": "volume_ratio_20d",
            "python_function": volume_ratio,
            "source": "custom_factor",
            "description": "20日成交量比率",
            "tags": ["量价类", "自定义"],
        },
        {
            "name": "bb_width",
            "python_function": bollinger_width,
            "source": "custom_factor",
            "description": "布林带宽度",
            "tags": ["波动率", "自定义"],
        },
    ]

    # 创建handler
    handler = CustomHandler(
        instruments=["SH600000", "SH600036", "SH600519"],
        start_time="2023-01-01",
        end_time="2023-12-31",
        factors=factors,
    )

    return handler


def load_factors_from_yaml(yaml_file_path):
    """
    从YAML文件加载因子配置

    Parameters:
    -----------
    yaml_file_path : str
        factors.yaml文件路径

    Returns:
    --------
    list : 因子配置列表
    """
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            factors = yaml.safe_load(f)
        return factors if isinstance(factors, list) else []
    except Exception as e:
        warnings.warn(f"Failed to load factors from {yaml_file_path}: {e}")
        return []


def create_handler_from_yaml(
    yaml_file_path, instruments="csi300", start_time=None, end_time=None, **kwargs
):
    """
    从YAML文件创建CustomHandler

    Parameters:
    -----------
    yaml_file_path : str
        factors.yaml文件路径
    instruments : str or list
        股票池
    start_time/end_time : str
        时间范围
    **kwargs :
        其他CustomHandler参数

    Returns:
    --------
    CustomHandler : 配置好的handler实例
    """
    factors = load_factors_from_yaml(yaml_file_path)

    return CustomHandler(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        factors=factors,
        **kwargs,
    )


def debug_handler():
    """调试TA-Lib计算问题"""
    try:
        # 初始化qlib
        import qlib
        from qlib.config import REG_CN

        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

        # 创建简化的测试因子
        test_factors = [
            {
                "name": "RSI14",
                "talib_function": "RSI",
                "params": {"input_col": "close", "timeperiod": 14},
                "source": "TA-Lib",
                "description": "14日相对强弱指标",
            },
            {
                "name": "ADX14",
                "talib_function": "ADX",
                "inputs": ["$high", "$low", "$close"],
                "params": {"timeperiod": 14},
                "source": "TA-Lib",
                "description": "14日平均趋向指标",
            },
        ]

        # 创建handler
        handler = CustomHandler(
            instruments=["SH600000"],  # 只测试一只股票
            start_time="2023-01-01",
            end_time="2023-03-31",  # 缩短时间范围
            factors=test_factors,
        )

        print("开始获取数据...")
        df = handler.fetch(selector=slice("2023-01-01", "2023-03-31"))

        print(f"\n数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        # 打印基础数据样本
        print("\n基础数据样本:")
        base_cols = [
            col
            for col in df.columns
            if col.startswith("$") or col in ["open", "high", "low", "close", "volume"]
        ]
        if base_cols:
            print(df[base_cols].head(10))

        # 检查TA-Lib指标
        talib_cols = [col for col in df.columns if col in ["RSI14", "ADX14"]]
        if talib_cols:
            print(f"\nTA-Lib指标:")
            for col in talib_cols:
                valid_count = df[col].count()
                total_count = len(df[col])
                print(f"{col}: {valid_count}/{total_count} 有效值")
                if valid_count > 0:
                    print(f"  范围: {df[col].min():.4f} - {df[col].max():.4f}")
                    print(f"  最后5个值: {df[col].tail().values}")
        print(df["RSI14"])
    except Exception as e:
        print(f"调试错误: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    # 使用示例
    print("CustomHandler TA-Lib 调试")
    print("=" * 50)

    debug_handler()
