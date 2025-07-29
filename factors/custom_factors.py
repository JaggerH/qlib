# custom_factors.py

import yaml
import os
from pathlib import Path


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
    custom_factors = {}
    for factor in factors_data:
        if factor.get("source") == "custom_factor":
            name = factor["name"]
            expression = factor["qlib_expression"]
            custom_factors[name] = expression

    return custom_factors


# 从YAML文件加载自定义因子
expressions = load_custom_factors_from_yaml()

from qlib.data.dataset.loader import QlibDataLoader


class CustomFactorDL(QlibDataLoader):
    """Dataloader to get CustomFactor"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config(
        config={
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
    ):
        fields = list(expressions.values())
        names = list(expressions.keys())

        return fields, names


from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class CustomFactor(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return CustomFactorDL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class CustomFactorvwap(CustomFactor):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]
