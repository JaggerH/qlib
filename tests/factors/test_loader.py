#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loader测试用例

测试内容：
1. 验证从factors.yaml中读取的source为custom_factor的因子数和输出的因子数保持一致
2. 输出的df必须是多级列索引，并且从df.columns获取到的因子要保证是('feature', 'factor_name')的格式
3. 输出的结果中不能包含OHLCV等基础数据
4. CIntradayDL测试：验证1min数据加载和按日期分组功能
"""

from re import T
import unittest
from qlib.data.dataset.loader import NestedDataLoader
import pandas as pd
import numpy as np

import qlib
from qlib.config import REG_CN

qlib.init(
    provider_uri={
        "day": "~/.qlib/qlib_data/cn_data",  # 日线数据
        "1min": "~/.qlib/qlib_data/cn_data_1min",  # 1分钟数据
    },
    region=REG_CN,
)

from factors.factor_inspector import read_corr_params
from factors.loader import CQilbDL, CPyDL, CIntradayDL


class TestLoader(unittest.TestCase):

    def test_CQilbDL(self):
        names, fields = CQilbDL.get_feature_config()
        self.assertEqual(len(names), len(fields))

        instruments, start_time, end_time = read_corr_params()

        nd = NestedDataLoader(
            dataloader_l=[
                {
                    "class": "factors.loader.CQilbDL",
                },
            ]
        )

        df = nd.load(instruments, start_time, end_time)
        for column in df.columns:
            self.assertEqual(column[0], "feature")

        self.assertEqual(len(df.columns), len(names))

    def test_CPyDL(self):
        fileds, names = CPyDL.get_feature_config()
        self.assertIsInstance(names, list)

        instruments, start_time, end_time = read_corr_params()

        nd = NestedDataLoader(
            dataloader_l=[
                {
                    "class": "factors.loader.CPyDL",
                },
            ]
        )

        df = nd.load(instruments, start_time, end_time)
        for column in df.columns:
            self.assertEqual(column[0], "feature")

        self.assertEqual(len(df.columns), len(names))

    def test_CIntradayDL_base_data_loading(self):
        """测试CIntradayDL的基础数据加载功能"""
        # 确保在测试过程中保持正确的1min数据配置
        import qlib

        qlib.init(
            provider_uri={
                "day": "~/.qlib/qlib_data/cn_data",  # 日线数据
                "1min": "~/.qlib/qlib_data/cn_data_1min",  # 1分钟数据
            },
            region="cn",
        )

        # 使用1min数据实际存在的时间范围和股票
        instruments, start_time, end_time = read_corr_params(use_1min=True)
        intraday_dl = CIntradayDL()
        base_data = intraday_dl._load_base_data(
            instruments, start_time, end_time, freq="1min"
        )
        # 判断base_data是否为DataFrameGroupBy对象
        self.assertIsInstance(base_data, pd.core.groupby.generic.DataFrameGroupBy)
        for date, df in base_data:
            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue(df.shape[0] >= 1, "基础数据行数应大等于1")
            self.assertTrue(df.shape[1] >= 1, "基础数据列数应大等于1")

    def test_CIntradayDL(self):
        qlib.init(
            provider_uri={
                "day": "~/.qlib/qlib_data/cn_data",  # 日线数据
                "1min": "~/.qlib/qlib_data/cn_data_1min",  # 1分钟数据
            },
            region="cn",
        )
        fileds, names = CIntradayDL.get_feature_config()
        self.assertIsInstance(names, list)

        instruments, start_time, end_time = read_corr_params(use_1min=True)

        nd = NestedDataLoader(
            dataloader_l=[
                {
                    "class": "factors.loader.CIntradayDL",
                },
            ]
        )

        df = nd.load(instruments, start_time, end_time)
        for column in df.columns:
            self.assertEqual(column[0], "feature")

        self.assertEqual(len(df.columns), len(names))


if __name__ == "__main__":
    unittest.main()
