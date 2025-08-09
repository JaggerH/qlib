import os
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


class CPyDL(DLWParser):
    """Python Code DataLoader for custom factors with python_code"""

    def __init__(self, config=None, **kwargs):
        self.yaml_path = kwargs.pop("yaml_path", None)
        # 解析YAML配置，获取所有python_code因子
        self.factor_configs = self._get_python_code_factors(self.yaml_path)

        # 编译所有因子代码
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

    def _get_python_code_factors(self, yaml_path=None):
        """获取所有python_code因子的通用函数"""
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "factors.yaml"
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            factors = yaml.safe_load(f)

        # 只保留source为"custom_factor"且包含"python_code"字段的因子
        python_code_factors = [
            factor
            for factor in factors
            if factor.get("source") == "custom_factor"
            and "python_code" in factor
            and "type" not in factor
        ]

        return python_code_factors

    def _compile_all_factors(self):
        """编译所有因子的Python代码"""
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

            # 为每个输出创建一个表达式和名称
            for output_name in outputs:
                all_exprs.append(f"{factor_name}_{output_name}")
                all_names.append(output_name)

        return (all_exprs, all_names)

    @staticmethod
    def get_feature_config(yaml_path=None):
        """获取CPyDL的特征配置"""
        # 创建临时实例来调用通用函数
        temp_instance = CPyDL.__new__(CPyDL)
        python_code_factors = temp_instance._get_python_code_factors(yaml_path)

        # 提取所有输出名称
        all_names = []
        for factor in python_code_factors:
            outputs = factor.get("outputs", ["result"])
            all_names.extend(outputs)

        return (all_names, all_names)

    def _load_base_data(self, instruments, start_time, end_time):
        """一次性加载所有基础数据"""
        # 收集所有因子需要的输入字段
        all_inputs = self._collect_all_factor_inputs()
        # 如果all_inputs为空，则直接返回空的DataFrame
        if not all_inputs:
            return pd.DataFrame()

        # 处理instruments参数：如果是字符串，转换为D.instruments格式
        if isinstance(instruments, str):
            instruments = D.instruments(market=instruments)

        # 一次性加载所有基础数据
        base_df = D.features(instruments, all_inputs, start_time, end_time, freq="day")

        return base_df

    def _collect_all_factor_inputs(self):
        """收集所有因子需要的输入字段"""
        all_inputs = set()
        for factor_config in self.factor_configs:
            inputs = factor_config.get("inputs", [])
            all_inputs.update(inputs)

        return list(all_inputs)

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
            # 从expr中提取因子名称（格式：{factor_name}_{output_name}）
            suffix = f"_{name}"
            if expr.endswith(suffix):
                factor_name = expr[: -len(suffix)]
            else:
                # 如果没有找到预期的后缀，使用原来的逻辑作为后备
                factor_name = expr.split("_")[0]

            # 判断是否为自定义因子
            if self._is_custom_factor(factor_name):
                # 如果因子还没处理过，执行计算
                if factor_name not in processed_factors:
                    current_factor_config = self._get_factor_config_by_name(factor_name)
                    all_results = self._execute_factor_with_base_data(
                        current_factor_config, self._base_data
                    )

                    # 将结果按输出名称分组
                    outputs = current_factor_config.get("outputs", ["result"])
                    for output_name in outputs:
                        if output_name in all_results:
                            factor_results.append(all_results[output_name])
                            factor_names.append(output_name)
                        else:
                            raise ValueError(
                                f"Output '{output_name}' not found in factor {factor_name}"
                            )

                    processed_factors.add(factor_name)
            else:
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

        # 3. 构建结果DataFrame
        result_df = self._build_factor_results_df(factor_results, factor_names)

        return result_df

    def _get_factor_config_by_name(self, factor_name):
        """根据因子名称获取因子配置"""
        for factor_config in self.factor_configs:
            if factor_config["name"] == factor_name:
                return factor_config
        raise ValueError(f"Factor {factor_name} not found in configuration")

    def _execute_factor_with_base_data(self, factor_config, base_data):
        """使用预加载的基础数据执行单个因子计算"""
        # 1. 提取因子需要的输入数据
        inputs = factor_config.get("inputs", [])
        factor_input_data = base_data[inputs]

        # 2. 准备Python代码执行环境
        exec_env = self._prepare_execution_environment(factor_input_data)

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

    def _prepare_execution_environment(self, factor_input_data):
        """准备因子Python代码的执行环境"""
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

    def _is_custom_factor(self, factor_name):
        """判断是否为自定义因子"""
        try:
            self._get_factor_config_by_name(factor_name)
            return True
        except ValueError:
            return False

    def _is_qlib_expression(self, expr):
        try:
            # 使用qlib官方的表达式验证方法
            ExpressionD.get_expression_instance(expr)
            return True
        except (NameError, SyntaxError, ValueError):
            # 如果qlib无法解析该表达式，则不是qlib表达式
            return False

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
        # 使用qlib的D.features方法计算表达式
        if instruments is None:
            warnings.warn("`instruments` is not set, will load all stocks")
            instruments = "all"
        if isinstance(instruments, str):
            instruments = D.instruments(instruments)
        # CPyDL处理日频数据，所以使用day频率
        result_df = D.features(instruments, [expr], start_time, end_time, freq="day")
        # 提取第一列作为Series
        result = result_df.iloc[:, 0]
        return result


class CIntradayDL(CPyDL):
    """日内数据加载器，支持自定义因子和标准qlib表达式"""

    def __init__(self, config=None, **kwargs):
        self.yaml_path = kwargs["yaml_path"]
        # 添加RTH过滤参数
        self.use_RTH = kwargs.pop("use_RTH", True)

        # 解析YAML配置，获取所有python_code因子
        self.factor_configs = self._get_python_code_factors(self.yaml_path)

        # 编译所有因子代码
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

    def _get_python_code_factors(self, yaml_path=None):
        """获取所有python_code因子的通用函数"""
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "factors.yaml"
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            factors = yaml.safe_load(f)

        # 只保留source为"custom_factor"且包含"python_code"字段的因子
        python_code_factors = [
            factor
            for factor in factors
            if factor.get("source") == "custom_factor"
            and factor.get("type") == "intraday_python_code"
        ]

        return python_code_factors

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
        all_inputs = self._collect_all_factor_inputs()
        if not all_inputs:
            return pd.DataFrame()
        if instruments is None:
            warnings.warn("`instruments` is not set, will load all stocks")
            instruments = "all"
        if isinstance(instruments, str):
            instruments = D.instruments(instruments)

        # 一次性加载所有基础数据
        base_df = D.features(instruments, all_inputs, start_time, end_time, freq=freq)

        # 检查数据是否为空
        if base_df.empty:
            raise ValueError(
                f"数据为空！请检查以下参数：\n"
                f"- instruments: {instruments}\n"
                f"- start_time: {start_time}\n"
                f"- end_time: {end_time}\n"
                f"- freq: {freq}\n"
                f"- all_inputs: {all_inputs}\n"
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
                # 处理自定义因子
                # 从expr中提取因子名称（格式：{factor_name}_{output_name}）
                suffix = f"_{name}"
                if expr.endswith(suffix):
                    factor_name = expr[: -len(suffix)]
                else:
                    # 如果没有找到预期的后缀，使用原来的逻辑作为后备
                    factor_name = expr.split("_")[0]

                # 检查因子是否存在
                # if not self._is_custom_factor(factor_name):
                #     raise ValueError(
                #         f"Custom factor '{factor_name}' not found in configuration"
                #     )

                # 如果因子还没处理过，执行计算
                if factor_name not in processed_factors:
                    current_factor_config = self._get_factor_config_by_name(factor_name)
                    all_results = self._execute_factor_with_base_data(
                        current_factor_config, self._base_data
                    )

                    # 将结果按输出名称分组
                    outputs = current_factor_config.get("outputs", ["result"])
                    for output_name in outputs:
                        if output_name in all_results:
                            factor_results.append(all_results[output_name])
                            # 使用因子名称而不是输出名称
                            factor_names.append(factor_name)
                        else:
                            raise ValueError(
                                f"Output '{output_name}' not found in factor {factor_name}"
                            )

                    processed_factors.add(factor_name)

        # 3. 构建结果DataFrame
        result_df = self._build_factor_results_df(factor_results, factor_names)

        return result_df

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
        python_code = factor_config.get("python_code", "")
        inputs = factor_config.get("inputs", [])
        outputs = factor_config.get("outputs", ["result"])
        factor_name = factor_config.get("name", "unknown")

        # 2. 初始化结果收集器
        results = {output: [] for output in outputs}
        instruments_list = []
        dates_list = []

        # 3. 核心计算循环：按股票和日期处理
        for (instrument, date), df in base_data:
            # 准备当前股票当前日期的输入数据
            local_vars = self._prepare_intraday_input_data(df, inputs)
            # 执行python_code计算
            exec_result = self._execute_intraday_python_code(
                python_code, local_vars, outputs
            )

            # 收集结果
            for output in outputs:
                if output in exec_result:
                    results[output].append(exec_result[output])
                else:
                    raise ValueError(
                        f"Output '{output}' not found in factor {factor_name} for {instrument} on {date}"
                    )
            instruments_list.append(instrument)
            dates_list.append(date)

        # 4. 结果转换：构建MultiIndex结果
        final_results = {}
        for output in outputs:
            if len(results[output]) > 0:
                # 确保datetime是pd.Timestamp类型，以便与字符串进行比较
                datetime_timestamps = [
                    pd.Timestamp(date) if hasattr(date, "date") else date
                    for date in dates_list
                ]
                # 创建MultiIndex
                multi_index = pd.MultiIndex.from_arrays(
                    [instruments_list, datetime_timestamps],
                    names=["instrument", "datetime"],
                )
                # 创建Series，使用MultiIndex
                final_results[output] = pd.Series(
                    results[output], index=multi_index, name=output
                )
            else:
                raise ValueError(
                    f"No results generated for output '{output}' in factor {factor_name}"
                )

        return final_results

    def _prepare_intraday_input_data(self, df, inputs):
        """
        为单日数据准备输入变量

        Args:
            df: 单日的分钟数据DataFrame
            inputs: 输入字段列表

        Returns:
            dict: 包含输入变量的字典
        """
        local_vars = {}

        # 输入字段映射
        column_map = {
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
            "$vwap": "vwap",
        }

        for input_name in inputs:
            if input_name in column_map:
                local_vars[column_map[input_name]] = df[input_name]
            else:
                local_vars[input_name] = df[input_name]

        return local_vars

    @staticmethod
    def get_feature_config(yaml_path=None):
        """获取CIntradayDL的特征配置"""
        # 创建临时实例来调用通用函数
        temp_instance = CIntradayDL.__new__(CIntradayDL)
        python_code_factors = temp_instance._get_python_code_factors(yaml_path)

        # 提取所有输出名称
        all_names = []
        for factor in python_code_factors:
            outputs = factor.get("outputs", ["result"])
            all_names.extend(outputs)

        return (all_names, all_names)

    def _execute_intraday_python_code(self, python_code, local_vars, outputs):
        """
        执行单日的python代码

        Args:
            python_code: 要执行的Python代码字符串
            local_vars: 局部变量字典
            outputs: 期望的输出变量列表

        Returns:
            dict: 包含输出变量的字典
        """
        # 添加必要的库导入
        local_vars.update(
            {
                "np": np,
                "pd": pd,
            }
        )

        # 尝试添加talib（如果可用）
        try:
            import talib

            local_vars["talib"] = talib
        except ImportError:
            pass

        # 执行代码
        exec(python_code, globals(), local_vars)

        # 收集输出结果
        result = {}
        for output in outputs:
            if output in local_vars:
                result[output] = local_vars[output]
            else:
                raise ValueError(
                    f"Output variable '{output}' not found after executing code"
                )

        return result

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
