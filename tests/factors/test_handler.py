#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
handler测试用例

handler主要用于workflow中DataSet的加载

测试内容：
1. 验证从factors.yaml中读取的source为custom_factor的因子数和输出的因子数保持一致
2. 输出的df必须是多级列索引，并且从df.columns获取到的因子要保证是('feature', 'factor_name')的格式
3. 输出的结果中不能包含OHLCV等基础数据
"""

import unittest

import qlib
from qlib.config import REG_CN

# qlib.init(region=REG_CN)

from qlib.utils import init_instance_by_config


class TestLoader(unittest.TestCase):

    def test_CombineHandler(self):
        qlib.init(
            provider_uri={
                "day": "~/.qlib/qlib_data/cn_data",  # 日线数据
                "1min": "~/.qlib/qlib_data/cn_data_1min",  # 1分钟数据
            },
            region=REG_CN,
        )
        market = "csi300"

        data_handler_config = {
            "start_time": "2020-09-15",
            "end_time": "2020-10-01",
            "fit_start_time": "2020-09-15",
            "fit_end_time": "2020-10-01",
            "instruments": market,
        }

        dataset = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "CombineHandler",
                    "module_path": "factors.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2020-09-15", "2020-10-01"),
                    "valid": ("2020-10-02", "2020-10-15"),
                    "test": ("2020-10-16", "2020-10-31"),
                },
            },
        }

        dataset = init_instance_by_config(dataset)

    def test_IntradayHandler(self):
        qlib.init(
            provider_uri={
                "day": "~/.qlib/qlib_data/cn_data",  # 日线数据
                "1min": "~/.qlib/qlib_data/cn_data_1min",  # 1分钟数据
            },
            region=REG_CN,
        )
        market = "csi300"

        data_handler_config = {
            "start_time": "2020-09-15",
            "end_time": "2020-10-01",
            "fit_start_time": "2020-09-15",
            "fit_end_time": "2020-10-01",
            "instruments": market,
        }

        dataset = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "IntradayHandler",
                    "module_path": "factors.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2020-09-15", "2020-10-01"),
                    "valid": ("2020-10-02", "2020-10-15"),
                    "test": ("2020-10-16", "2020-10-31"),
                },
            },
        }

        dataset = init_instance_by_config(dataset)

        # 准备数据
        df = dataset.prepare("train")
        print(df)

if __name__ == "__main__":
    unittest.main()
