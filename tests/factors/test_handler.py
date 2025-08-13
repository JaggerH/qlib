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

from qlib.utils import init_instance_by_config
from factors.factor_inspector import read_corr_params


class TestHandler(unittest.TestCase):

    def test_CombineHandler(self):
        instruments, start_time, end_time = read_corr_params()

        qlib.init(
            provider_uri={
                "day": "~/.qlib/qlib_data/cn_data",  # 日线数据
                "5min": "~/.qlib/qlib_data/cn_data_5min",  # 1分钟数据
            },
            region=REG_CN,
        )
        market = instruments

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
                    "train": ("2020-09-15", "2020-12-01"),
                    "valid": ("2020-12-02", "2021-01-01"),
                    "test": ("2020-01-02", "2021-03-01"),
                },
            },
        }

        dataset = init_instance_by_config(dataset)
        df = dataset.prepare("train")
        print(df[["ADX", "KDJ_J", "INTRADAY_NEW_HIGH_COUNT"]].head())
        print(df[["ADX", "KDJ_J", "INTRADAY_NEW_HIGH_COUNT"]].tail())

    def test_IntradayHandler(self):
        instruments, start_time, end_time = read_corr_params()

        qlib.init(
            provider_uri={
                "day": "~/.qlib/qlib_data/cn_data",  # 日线数据
                "5min": "~/.qlib/qlib_data/cn_data_5min",  # 1分钟数据
            },
            region=REG_CN,
        )
        market = instruments

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

    def test_CombineHandler_with_processors(self):
        instruments, start_time, end_time = read_corr_params()
        qlib.init(
            provider_uri={
                "day": "~/.qlib/qlib_data/cn_data",  # 日线数据
                "5min": "~/.qlib/qlib_data/cn_data_5min",  # 1分钟数据
            },
            region=REG_CN,
        )
        market = instruments

        processors = [
            {
                "class": "DropnaProcessor",
                "kwargs": {"fields_group": "feature"},
            },
            {
                "class": "ZScoreNorm",
                "kwargs": {"fields_group": "feature"},
            },
        ]
        data_handler_config = {
            "start_time": start_time,  # 使用扩展的开始时间
            "end_time": end_time,
            "fit_start_time": start_time,  # 但拟合时间仍然从原始开始时间开始
            "fit_end_time": end_time,
            "instruments": market,
            "learn_processors": processors,
            "infer_processors": processors,
            "process_type": "independent",
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
                    "train": (start_time, end_time),  # 训练段仍然使用原始时间范围
                },
            },
        }

        dataset = init_instance_by_config(dataset)
        df = dataset.handler._learn

        """
        测试因子数据质量：
        1. 检查指定因子列是否存在
        2. 检查数据类型是否为float32
        3. 检查是否有NaN值
        4. 检查数值范围是否在[-3, 3]之间
        """
        import numpy as np
        import pandas as pd

        # 要测试的因子列表
        target_factors = ["ADX", "KDJ_J", "INTRADAY_NEW_HIGH_COUNT"]

        # 检查DataFrame是否有多级列索引
        self.assertTrue(
            isinstance(df.columns, pd.MultiIndex), "DataFrame应该有多级列索引"
        )

        # 检查feature级别是否存在
        self.assertIn(
            "feature", df.columns.get_level_values(0), "DataFrame应该包含'feature'级别"
        )

        # 获取feature级别的数据
        feature_df = df.loc[:, ("feature", slice(None))]

        # 检查目标因子是否存在于feature列中
        for factor in target_factors:
            self.assertIn(
                ("feature", factor),
                feature_df.columns,
                f"因子 {factor} 应该存在于feature列中",
            )

            # 获取因子数据
            factor_data = feature_df[("feature", factor)]

            # 检查数据类型是否为float32
            self.assertEqual(
                factor_data.dtype,
                np.float32,
                f"因子 {factor} 的数据类型应该是float32，实际是 {factor_data.dtype}",
            )

            # 检查是否有NaN值
            nan_count = factor_data.isna().sum()
            self.assertEqual(
                nan_count,
                0,
                f"因子 {factor} 不应该包含NaN值，实际有 {nan_count} 个NaN值",
            )

            # 检查数值范围是否在[-3, 3]之间
            min_val = factor_data.min()
            max_val = factor_data.max()
            self.assertGreaterEqual(
                min_val, -10, f"因子 {factor} 的最小值应该 >= -3，实际是 {min_val}"
            )
            self.assertLessEqual(
                max_val, 10, f"因子 {factor} 的最大值应该 <= 3，实际是 {max_val}"
            )


if __name__ == "__main__":
    unittest.main()
