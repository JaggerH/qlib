import os
import json
import yaml
import warnings
import pandas as pd
import numpy as np
from qlib.data.dataset.loader import QlibDataLoader, DLWParser
from qlib.data import D
from qlib.data.data import ExpressionD


class CQilbDL(QlibDataLoader):
    """Dataloader to get CQilb"""

    def __init__(self, config=None, **kwargs):
        self.yaml_path = kwargs.pop("yaml_path", None)
        _config = {
            "feature": self.get_feature_config(self.yaml_path),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config(yaml_path=None):
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "factors.yaml"
            )
        with open(yaml_path, "r", encoding="utf-8") as f:
            factors = yaml.safe_load(f)

        # 只保留source为"custom_factor"且包含"qlib_expression"字段的因子
        factors = [
            {
                "name": factor["name"],
                "field": factor["qlib_expression"],
            }
            for factor in factors
            if factor.get("source") == "custom_factor" and "qlib_expression" in factor
        ]

        names = [factor["name"] for factor in factors]
        fields = [factor["field"] for factor in factors]
        return fields, names


class BasePyDL(DLWParser):
    """Python Code DataLoader 基础抽象类，用于自定义因子"""

    def __init__(self, config=None, **kwargs):
        default_yaml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "factors.yaml"
        )
        yaml_path = kwargs.pop("yaml_path", None)
        self.yaml_path = yaml_path if yaml_path is not None else default_yaml_path

        # 过滤并编译因子代码
        self._compile_all_factors()

        # 构建DLWParser配置
        _config = {
            "feature": self._build_dlwparser_config(),
        }
        if config is not None:
            _config.update(config)
        # 初始化基础数据缓存
        self._base_data = None

        # 调用父类初始化
        super().__init__(config=_config, **kwargs)

    def _compile_all_factors(self):
        """
        获取所有python_code因子的通用函数
        编译所有因子的Python代码
        """
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            factors = yaml.safe_load(f)

        # 子类需要重写此方法来定义具体的因子过滤逻辑
        self.factor_configs = self._filter_factors(factors)
        self.compiled_factors = {}

        for factor_config in self.factor_configs:
            factor_name = factor_config["name"]
            python_code = factor_config["python_code"]

            try:
                # 编译Python代码
                compiled_code = compile(python_code, f"<{factor_name}>", "exec")
                self.compiled_factors[factor_name] = compiled_code
            except Exception as e:
                raise ValueError(f"Failed to compile factor {factor_name}: {e}")

    def _build_dlwparser_config(self):
        """构建DLWParser期望的配置格式"""
        # 提取所有因子的输出名称
        all_names = []
        all_exprs = []

        for factor_config in self.factor_configs:
            factor_name = factor_config["name"]
            outputs = factor_config.get("outputs", ["result"])

            if len(outputs) == 1:
                all_exprs.append(json.dumps(factor_config))
                all_names.append(factor_name)
            else:
                for output_name in outputs:
                    all_exprs.append(json.dumps(factor_config))
                    all_names.append(output_name)

        return (all_exprs, all_names)

    def _collect_all_factor_inputs(self):
        """收集所有因子需要的输入字段"""
        all_inputs = set()
        for factor_config in self.factor_configs:
            inputs = factor_config.get("inputs", [])
            all_inputs.update(inputs)

        return list(all_inputs)

    def _get_factor_config_by_name(self, factor_name):
        """根据因子名称获取因子配置"""
        for factor_config in self.factor_configs:
            if factor_config["name"] == factor_name:
                return factor_config
        raise ValueError(f"Factor {factor_name} not found in configuration")

    def _is_qlib_expression(self, expr):
        return ExpressionD.check_expression(expr)

    def _prepare_instruments(self, instruments):
        """准备股票代码列表"""
        if instruments is None:
            warnings.warn("`instruments` is not set, will load all stocks")
            instruments = "all"
        if isinstance(instruments, str):
            instruments = D.instruments(instruments)
        return instruments

    def _calculate_qlib_expression(self, expr, instruments, start_time, end_time):
        """
        计算标准的qlib表达式（如标签表达式）

        Args:
            expr: qlib表达式字符串
            instruments: 股票代码列表
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            Series: 计算结果
        """
        instruments = self._prepare_instruments(instruments)
        # 子类需要重写此方法来指定正确的频率
        freq = self._get_qlib_expression_freq()
        result_df = D.features(instruments, [expr], start_time, end_time, freq=freq)
        # 提取第一列作为Series
        result = result_df.iloc[:, 0]
        return result

    def _get_qlib_expression_freq(self):
        """获取qlib表达式的频率，子类需要重写"""
        raise NotImplementedError("Subclasses must implement _get_qlib_expression_freq")

    def _get_factor_name_from_expr(self, expr, name):
        """
        从表达式中提取因子名称
        expr = {factor_name}_{output_name}
        """
        factor_config = json.loads(expr)
        factor_name = factor_config["name"]
        return factor_name

    def _load_base_data(self, instruments, start_time, end_time, freq="day"):
        """一次性加载所有基础数据"""
        # 收集所有因子需要的输入字段
        all_inputs = self._collect_all_factor_inputs()
        # 如果all_inputs为空，则直接返回空的DataFrame
        if not all_inputs:
            return pd.DataFrame()

        instruments = self._prepare_instruments(instruments)
        # 一次性加载所有基础数据
        base_df = D.features(instruments, all_inputs, start_time, end_time, freq=freq)

        return base_df

    def load_group_df(
        self, instruments, exprs, names, start_time=None, end_time=None, gp_name=None
    ):
        """处理单个因子组的数据加载和计算"""
        # 1. 加载基础数据（如果还没加载）
        if self._base_data is None:
            self._base_data = self._load_base_data(instruments, start_time, end_time)

        # 2. 执行因子计算
        factor_results = []
        factor_names = []

        # 按因子分组处理
        processed_factors = set()

        for expr, name in zip(exprs, names):
            # 判断表达式是否为qlib表达式
            if self._is_qlib_expression(expr):
                # 处理标准qlib表达式（如标签表达式）
                try:
                    # 使用qlib的标准表达式计算
                    result = self._calculate_qlib_expression(
                        expr, instruments, start_time, end_time
                    )
                    factor_results.append(result)
                    factor_names.append(name)
                except Exception as e:
                    raise ValueError(
                        f"Failed to calculate qlib expression '{expr}': {str(e)}"
                    )
            else:
                factor_name = self._get_factor_name_from_expr(expr, name)

                # 如果因子还没处理过，执行计算
                if factor_name not in processed_factors:
                    current_factor_config = self._get_factor_config_by_name(factor_name)
                    all_results = self._execute_factor_with_base_data(
                        current_factor_config, self._base_data
                    )

                    # 将结果按输出名称分组
                    outputs = current_factor_config.get("outputs", ["result"])
                    if len(outputs) == 1:
                        factor_results.append(all_results[outputs[0]])
                        factor_names.append(factor_name)
                    else:
                        for output_name in outputs:
                            if output_name in all_results:
                                factor_results.append(all_results[output_name])
                                # 使用因子名称而不是输出名称
                                factor_names.append(output_name)
                            else:
                                raise ValueError(
                                    f"Output '{output_name}' not found in factor {factor_name}"
                                )

                    processed_factors.add(factor_name)

        # 3. 构建结果DataFrame
        result_df = self._build_factor_results_df(factor_results, factor_names)

        return result_df

    def _prepare_execution_environment(self, df, inputs):
        """准备因子Python代码的执行环境"""
        factor_input_data = df[inputs]
        # 将DataFrame的列转换为Series，作为变量传递给Python代码
        column_map = {
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
            "$vwap": "vwap",
        }
        exec_env = {}
        for col_name in factor_input_data.columns:
            if col_name in column_map.keys():
                exec_env[column_map[col_name]] = factor_input_data[col_name]
            else:
                exec_env[col_name] = factor_input_data[col_name]

        # 添加常用的库
        exec_env.update(
            {
                "pd": pd,
                "np": np,
            }
        )

        # 尝试添加talib（如果可用）
        try:
            import talib

            exec_env["talib"] = talib
        except ImportError:
            pass

        return exec_env

    def _execute_factor_with_base_data(self, factor_config, base_data):
        """执行因子计算的抽象方法，子类必须实现"""
        raise NotImplementedError(
            "Subclasses must implement _execute_factor_with_base_data"
        )

    def _build_factor_results_df(self, factor_results, names):
        """构建因子结果DataFrame的抽象方法，子类必须实现"""
        raise NotImplementedError("Subclasses must implement _build_factor_results_df")

    def _filter_factors(self, factors):
        """过滤因子的抽象方法，子类必须实现"""
        raise NotImplementedError("Subclasses must implement _filter_factors")

    @staticmethod
    def get_feature_config(yaml_path=None):
        """获取特征配置的抽象方法，子类需要实现"""
        raise NotImplementedError("Subclasses must implement get_feature_config")


class CPyDL(BasePyDL):
    """Python Code DataLoader for custom factors with python_code (日频数据)"""

    def _filter_factors(self, factors):
        """过滤日频python_code因子"""
        # 只保留source为"custom_factor"且包含"python_code"字段的因子
        python_code_factors = [
            factor
            for factor in factors
            if factor.get("source") == "custom_factor"
            and "python_code" in factor
            and "type" not in factor
        ]
        return python_code_factors

    @staticmethod
    def get_feature_config(yaml_path=None):
        """获取CPyDL的特征配置"""
        # 创建临时实例来调用通用函数
        temp_instance = CPyDL(yaml_path=yaml_path)
        all_names = temp_instance.fields["feature"][0]

        return (all_names, all_names)

    def _execute_factor_with_base_data(self, factor_config, base_data):
        """使用预加载的基础数据执行单个因子计算"""
        # 1. 准备基础配置
        inputs = factor_config.get("inputs", [])

        # 2. 准备Python代码执行环境
        exec_env = self._prepare_execution_environment(base_data, inputs)

        # 3. 执行因子代码
        factor_name = factor_config["name"]
        compiled_code = self.compiled_factors[factor_name]

        # 执行代码
        exec(compiled_code, exec_env)

        # 4. 获取所有输出结果
        outputs = factor_config.get("outputs", ["result"])
        results = {}

        for output_name in outputs:
            if output_name not in exec_env:
                raise ValueError(
                    f"Factor {factor_name} did not produce a '{output_name}' variable"
                )

            result = exec_env[output_name]

            # 验证结果
            self._validate_factor_output(result, f"{factor_name}_{output_name}")

            results[output_name] = result

        return results

    def _validate_factor_output(self, result, factor_name):
        """验证因子输出结果的格式"""
        if not isinstance(result, (pd.Series, pd.DataFrame)):
            raise ValueError(
                f"Factor {factor_name} must return a pandas.Series or pandas.DataFrame"
            )

        if isinstance(result, pd.Series):
            if set(result.index.names) != {"datetime", "instrument"}:
                raise ValueError(
                    f"Factor {factor_name} result must have MultiIndex with names ['datetime', 'instrument']"
                )

    def _get_qlib_expression_freq(self):
        """获取qlib表达式的频率 - 日频数据"""
        return "day"

    def _build_factor_results_df(self, factor_results, names):
        """构建因子结果DataFrame"""
        if not factor_results:
            # 返回一个空的DataFrame，但具有MultiIndex（instrument, datetime）
            index = pd.MultiIndex.from_arrays(
                [[], []], names=["instrument", "datetime"]
            )
            return pd.DataFrame(index=index)
        # 确保所有结果具有相同的索引
        base_index = factor_results[0].index

        # 构建结果DataFrame
        result_df = pd.DataFrame(
            {name: result for name, result in zip(names, factor_results)},
            index=base_index,
        )

        return result_df


class CIntradayDL(BasePyDL):
    """日内数据加载器，支持自定义因子和标准qlib表达式"""

    def __init__(self, config=None, **kwargs):
        # 添加RTH过滤参数
        self.use_RTH = kwargs.pop("use_RTH", True)
        super().__init__(config=config, **kwargs)

    def _filter_factors(self, factors):
        """过滤日内python_code因子"""
        # 只保留source为"custom_factor"且type为"intraday_python_code"的因子
        python_code_factors = [
            factor
            for factor in factors
            if factor.get("source") == "custom_factor"
            and factor.get("type") == "intraday_python_code"
        ]
        return python_code_factors

    @staticmethod
    def get_feature_config(yaml_path=None):
        """获取CIntradayDL的特征配置"""
        # 创建临时实例来调用通用函数
        temp_instance = CIntradayDL(yaml_path=yaml_path)
        all_names = temp_instance.fields["feature"][0]
        return (all_names, all_names)

    def _is_us_stock_minute_data(self):
        """
        判断是否为美股分钟数据

        Returns:
            bool: 是否为美股分钟数据
        """

        # 使用 qlib 的配置系统判断当前区域
        from qlib.config import C
        from qlib.constant import REG_US

        return C.region == REG_US

    def _filter_rth_data(self, df):
        """
        过滤RTH（Regular Trading Hours）数据

        Args:
            df: 包含datetime索引的DataFrame

        Returns:
            DataFrame: 过滤后的RTH数据
        """
        if df.empty:
            return df

        # 确保datetime是索引
        if "datetime" not in df.index.names:
            raise ValueError(
                "DataFrame must have 'datetime' in index names for RTH filtering"
            )

        # 获取datetime列
        datetime_index = df.index.get_level_values("datetime")

        # 转换为pandas datetime（如果不是的话）
        if not pd.api.types.is_datetime64_any_dtype(datetime_index):
            datetime_index = pd.to_datetime(datetime_index)

        # 提取时间部分
        time_only = datetime_index.time

        # 美股RTH时间：9:30 AM - 4:00 PM ET
        rth_start = pd.Timestamp("09:30").time()
        rth_end = pd.Timestamp("16:00").time()

        # 创建RTH过滤条件
        rth_mask = (time_only >= rth_start) & (time_only <= rth_end)

        # 应用过滤
        filtered_df = df[rth_mask]

        return filtered_df

    def _load_base_data(self, instruments, start_time, end_time, freq="5min"):
        """一次性加载所有基础数据"""
        # 收集所有因子需要的输入字段
        base_df = super()._load_base_data(instruments, start_time, end_time, freq)

        # 检查数据是否为空
        if base_df.empty:
            raise ValueError(
                f"数据为空！请检查以下参数：\n"
                f"- instruments: {instruments}\n"
                f"- start_time: {start_time}\n"
                f"- end_time: {end_time}\n"
                f"- freq: {freq}\n"
                f"可能的原因：\n"
                f"1. 指定的时间范围内没有数据\n"
                f"2. 指定的股票代码不存在或已退市\n"
                f"3. 数据源中没有对应的字段数据"
            )

        # 检查是否为美股分钟数据且需要RTH过滤
        if self.use_RTH and self._is_us_stock_minute_data():
            base_df = self._filter_rth_data(base_df)

        # 按股票和日期分组：将同一股票同一日期的所有分钟数据聚合在一起
        # 从MultiIndex中提取股票代码和日期
        instruments = base_df.index.get_level_values("instrument")
        # 保持datetime为pd.Timestamp类型，不转换为date
        dates = base_df.index.get_level_values("datetime").normalize()

        # 使用股票和日期进行分组，保持原始的MultiIndex结构
        grouped_data = base_df.groupby([instruments, dates], group_keys=False)

        return grouped_data

    def _execute_factor_with_base_data(self, factor_config, base_data):
        """
        执行日内因子计算

        Args:
            factor_config: 因子配置字典
            base_data: 按日期分组的分钟数据 {date: DataFrame}

        Returns:
            dict: {output_name: DataFrame/Series} 格式的结果
        """
        # 1. 提取因子配置
        inputs = factor_config.get("inputs", [])
        outputs = factor_config.get("outputs", ["result"])
        factor_name = factor_config.get("name", "unknown")
        compiled_code = self.compiled_factors[factor_name]

        # 2. 初始化结果收集器
        results = []

        # 缺乏重命名

        # 3. 核心计算循环：按股票和日期处理
        for (instrument, date), df in base_data:
            # 3.1 准备Python代码执行环境
            exec_env = self._prepare_execution_environment(df, inputs)
            # 3.2 执行因子代码
            exec(compiled_code, exec_env)  # 这里得到的是单行数据

            row = {"instrument": instrument, "date": date}
            # 3.3 收集结果
            for output in outputs:
                if output in exec_env:
                    row[output] = exec_env[output]
                else:
                    raise ValueError(
                        f"Output '{output}' not found in factor {factor_name} for {instrument} on {date}"
                    )
            results.append(row)

        # 4. 结果转换：构建MultiIndex结果
        final_results = pd.DataFrame(results)
        final_results = final_results.set_index(["instrument", "date"])

        return final_results

    def _get_qlib_expression_freq(self):
        """获取qlib表达式的频率 - 日内数据"""
        return "day"

    def _build_factor_results_df(self, factor_results, names):
        """
        构建日内因子结果DataFrame

        Args:
            factor_results: 因子结果列表，每个元素是pd.Series，具有MultiIndex(instrument, datetime)
            names: 因子名称列表

        Returns:
            pd.DataFrame: 具有MultiIndex(instrument, datetime)的因子结果
        """
        # 检查输入
        if not factor_results:
            raise ValueError("factor_results cannot be empty")

        if len(factor_results) != len(names):
            raise ValueError("factor_results and names must have the same length")

        # 获取所有结果的索引并集
        all_indices = set()
        for result in factor_results:
            if isinstance(result, pd.Series):
                all_indices.update(result.index)
            else:
                raise ValueError(f"Expected pd.Series, got {type(result)}")

        # 创建统一的MultiIndex
        unified_index = pd.MultiIndex.from_tuples(
            sorted(all_indices), names=["instrument", "datetime"]
        )

        # 构建结果DataFrame
        result_df = pd.DataFrame(index=unified_index)

        # 添加每个因子的结果
        for name, result in zip(names, factor_results):
            result_reindexed = result.reindex(unified_index)
            result_df[name] = result_reindexed

        return result_df
