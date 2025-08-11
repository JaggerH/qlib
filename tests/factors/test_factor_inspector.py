import unittest
import pandas as pd
import numpy as np

from factors.factor_inspector import read_corr_params, generate_handler_csv, generate_factor_csv


class TestFactorInspector(unittest.TestCase):
    """测试factor_inspector模块的功能"""
    def test_read_corr_params(self):
        """测试read_corr_params函数"""
        # 测试正常情况
        instruments, start_time, end_time = read_corr_params()

        # 验证返回值不为空
        self.assertIsNotNone(instruments)
        self.assertIsNotNone(start_time)
        self.assertIsNotNone(end_time)

        # 验证返回值的类型和内容
        self.assertIsInstance(instruments, list)
        self.assertEqual(len(instruments), 1)

    def test_generate_handler_csv(self):
        df = generate_handler_csv()
        print(df)

    def test_generate_factor_csv(self):
        df = generate_factor_csv("./test.yaml")
        print(df)

if __name__ == "__main__":
    unittest.main()
