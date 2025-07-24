# custom_factors.py

# 定义因子表达式（使用 Qlib 表达式语法）
expressions = {
    # 0. 目标变量: 下期收益率
    "label": "Ref($close, -1) / Ref($close, 0) - 1",
    # 1. 动量因子: 5日涨跌幅
    "momentum_5": "Ref($close, 0) / Ref($close, 5) - 1",
    # 2. 成交量比: 5日均量/10日均量 - 1
    "vol_ratio": "Mean($volume, 5) / Mean($volume, 10) - 1",
    # 3. RSI_14: 14日相对强弱指标
    # U = EMA(max(price_change,0), 14)，D = EMA(max(-price_change,0),14)
    # RSI = 100 - 100 / (1 + U/D)
    "price_change": "$close - Ref($close, 1)",
    "gain": "If($close - Ref($close, 1) > 0, $close - Ref($close, 1), 0)",
    "loss": "If($close - Ref($close, 1) < 0, Ref($close, 1) - $close, 0)",
    "avg_gain": "EMA($gain, 14)",
    "avg_loss": "EMA($loss, 14)",
    "RSI_14": "100 - 100 / (1 + $avg_gain / $avg_loss)",
    # 4. 布林带 (Bollinger Bands) 中轨/上轨/下轨
    "BB_middle": "Mean($close, 20)",
    "BB_upper": "Mean($close, 20) + 2 * Std($close, 20)",
    "BB_lower": "Mean($close, 20) - 2 * Std($close, 20)",
}

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
