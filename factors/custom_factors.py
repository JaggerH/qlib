# custom_factors.py

import yaml
import warnings
from pathlib import Path
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc

# 导入CustomHandler
try:
    from .custom_handler import CustomHandler
except ImportError:
    try:
        from factors.custom_handler import CustomHandler
    except ImportError:
        from custom_handler import CustomHandler


def load_custom_factors_from_yaml():
    """从factors.yaml文件中加载source为custom_factor的因子"""
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent
    yaml_file = current_dir / "factors.yaml"

    if not yaml_file.exists():
        raise FileNotFoundError(f"factors.yaml文件不存在: {yaml_file}")

    with open(yaml_file, "r", encoding="utf-8") as f:
        factors_data = yaml.safe_load(f)

    # 过滤出source为custom_factor的因子
    custom_factors = []
    for factor in factors_data:
        if factor.get("source") == "custom_factor":
            custom_factors.append(factor)

    return custom_factors


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class CustomFactor(CustomHandler):
    """
    整合的CustomFactor类，支持qlib表达式、TA-Lib和Python函数

    基于CustomHandler实现，从factors.yaml读取source为custom_factor的所有因子
    完全兼容qrun workflow
    """

    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=None,  # CustomHandler doesn't use process_type
        filter_pipe=None,
        inst_processors=None,
        factors_source="yaml",  # "yaml" 或自定义因子列表
        **kwargs,
    ):
        """
        Parameters:
        -----------
        instruments : str or list
            股票池
        start_time/end_time : str
            时间范围
        freq : str
            数据频率
        infer_processors : list
            推理时数据处理器（暂不支持）
        learn_processors : list
            训练时数据处理器（暂不支持）
        fit_start_time/fit_end_time : str
            拟合时间范围（暂不支持）
        process_type : str
            处理类型（暂不支持）
        filter_pipe : list
            过滤管道（暂不支持）
        inst_processors : list
            股票级别处理器（暂不支持）
        factors_source : str or list
            因子来源，"yaml"表示从factors.yaml读取，或直接传入因子列表
        """

        # 加载因子配置
        if factors_source == "yaml":
            factors = load_custom_factors_from_yaml()
        elif isinstance(factors_source, list):
            factors = factors_source
        else:
            raise ValueError(
                "factors_source must be 'yaml' or a list of factor configs"
            )

        # 直接调用CustomHandler的初始化
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            factors=factors,
            **kwargs,
        )

    def get_label_config(self):
        """默认标签配置"""
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    def get_feature_config(self):
        """
        获取因子配置，返回qlib表达式格式

        注意：这个方法主要用于与factor_inspector的兼容性
        对于TA-Lib和Python函数，返回占位符表达式

        Returns:
            tuple: (fields, names) - qlib表达式列表和因子名称列表
        """
        # 确保因子已解析
        if not hasattr(self, "factors") or not self.factors:
            factors = load_custom_factors_from_yaml()
        else:
            factors = self.factors

        fields = []
        names = []

        for factor in factors:
            name = factor.get("name")
            if not name:
                continue

            # qlib表达式类型 - 直接使用
            if "qlib_expression" in factor:
                fields.append(factor["qlib_expression"])
                names.append(name)

            # TA-Lib函数类型 - 使用占位符表达式
            elif "talib_function" in factor:
                # 为TA-Lib函数创建占位符表达式
                # 这些因子需要通过CustomFactor的计算管道来处理
                fields.append(f"$close")  # 占位符，实际计算由TA-Lib处理
                names.append(name)

            # Python代码类型 - 使用占位符表达式
            elif "python_code" in factor:
                # 为Python代码创建占位符表达式
                fields.append(f"$close")  # 占位符，实际计算由Python代码处理
                names.append(name)

        return fields, names

    def get_factor_names(self):
        """
        获取所有因子名称列表

        Returns:
            list: 因子名称列表
        """
        # 确保因子已解析
        if not hasattr(self, "factors") or not self.factors:
            factors = load_custom_factors_from_yaml()
        else:
            factors = self.factors

        names = []
        for factor in factors:
            name = factor.get("name")
            if name:
                names.append(name)

        return names

    def get_factor_configs_by_type(self):
        """
        按类型分组获取因子配置

        Returns:
            dict: {"qlib": [...], "talib": [...], "python": [...]}
        """
        # 确保因子已解析
        if not hasattr(self, "factors") or not self.factors:
            factors = load_custom_factors_from_yaml()
        else:
            factors = self.factors

        configs = {"qlib": {}, "talib": {}, "python": {}}

        for factor in factors:
            name = factor.get("name")
            if not name:
                continue

            if "qlib_expression" in factor:
                configs["qlib"][name] = {
                    "type": "qlib_expression",
                    "expression": factor["qlib_expression"],
                }
            elif "talib_function" in factor:
                configs["talib"][name] = {
                    "type": "talib_function",
                    "function": factor["talib_function"],
                    "parameters": factor.get("params", {}),
                }
            elif "python_code" in factor:
                configs["python"][name] = {
                    "type": "python_code",
                    "code": factor["python_code"],
                }

        return configs


class CustomFactorvwap(CustomFactor):
    """使用VWAP作为标签的CustomFactor"""

    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


# 保持向后兼容性的遗留类
class CustomFactorDL:
    """
    已废弃：请使用新的CustomFactor类

    保留此类仅为向后兼容，建议迁移到新的CustomFactor
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "CustomFactorDL is deprecated. Please use the new CustomFactor class which supports "
            "qlib expressions, TA-Lib functions, and Python functions.",
            DeprecationWarning,
            stacklevel=2,
        )


# 便利函数
def create_custom_factor_handler(
    instruments="csi500", start_time=None, end_time=None, **kwargs
):
    """
    便利函数：直接创建CustomFactor实例

    Returns:
    --------
    CustomFactor : 配置好的CustomFactor实例
    """
    return CustomFactor(
        instruments=instruments, start_time=start_time, end_time=end_time, **kwargs
    )


def get_custom_factors_dataframe(
    instruments="csi500", start_time=None, end_time=None, **kwargs
):
    """
    便利函数：直接获取自定义因子数据DataFrame

    Returns:
    --------
    pd.DataFrame : 因子数据
    """
    handler = create_custom_factor_handler(instruments, start_time, end_time, **kwargs)
    return handler.fetch()

