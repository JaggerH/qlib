import os
import yaml
import pandas as pd
import numpy as np
from qlib.data.dataset.loader import QlibDataLoader, DLWParser
from qlib.data import D


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
        self.factor_configs = self._parse_yaml_config(self.yaml_path)

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

    def _parse_yaml_config(self, yaml_path=None):
        """解析YAML配置文件，提取python_code因子"""
        return self._get_python_code_factors(yaml_path)

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
            if factor.get("source") == "custom_factor" and "python_code" in factor
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
            # 从expr中提取因子名称（格式：factor_name_output_name）
            factor_name = expr.split("_")[0]  # 取第一部分作为因子名

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
        # 确保所有结果具有相同的索引
        base_index = factor_results[0].index

        # 构建结果DataFrame
        result_df = pd.DataFrame(
            {name: result for name, result in zip(names, factor_results)},
            index=base_index,
        )

        return result_df
