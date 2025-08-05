#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loader测试用例

测试内容：
1. 验证从factors.yaml中读取的source为custom_factor的因子数和输出的因子数保持一致
2. 输出的df必须是多级列索引，并且从df.columns获取到的因子要保证是('feature', 'factor_name')的格式
3. 输出的结果中不能包含OHLCV等基础数据
"""

import unittest
from qlib.data.dataset.loader import NestedDataLoader

import qlib
from qlib.config import REG_CN

qlib.init(region=REG_CN)

from factors.factor_inspector import read_corr_params
from factors.loader import CQilbDL, CPyDL


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
        names = CPyDL.get_feature_config()
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


if __name__ == "__main__":
    unittest.main()
