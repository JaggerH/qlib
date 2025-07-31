import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

from unittest.mock import patch, MagicMock


# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from factors.factor_inspector import (
    FactorCalculator,
    read_corr_params,
    calculate_and_validate_factors,
)
import qlib
from qlib.config import REG_CN
from qlib.data import D


class TestFactorInspector(unittest.TestCase):
    """测试factor_inspector模块的功能"""

    def setUp(self):
        """测试前的准备工作"""
        # 使用qlib官方的日志控制方法
        from qlib.log import set_global_logger_level_cm
        import logging

        # 初始化qlib
        qlib.init(region=REG_CN, logging_level=logging.CRITICAL)

        # 使用上下文管理器设置全局日志级别为CRITICAL
        self.log_context = set_global_logger_level_cm(logging.CRITICAL)
        self.log_context.__enter__()

        # 创建临时配置文件
        self.temp_config = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        self.config_content = """
corr_params:
  instruments: ["000001.SZ", "000002.SZ"]
  start_time: "2023-01-01"
  end_time: "2023-01-31"
"""
        self.temp_config.write(self.config_content)
        self.temp_config.close()

        # 创建测试用的因子表达式
        self.test_expressions = {
            "qlib_factor": {"type": "qlib_expression", "expression": "$close / $open"},
            "talib_factor": {
                "type": "talib_function",
                "function": "SMA",
                "parameters": {"timeperiod": 5},
            },
            "python_factor": {"type": "python_code", "code": "result = close / open"},
        }

        # 创建测试用的基础数据
        self.df_base = self._create_test_data()

    def tearDown(self):
        """测试后的清理工作"""
        # 清理日志上下文
        if hasattr(self, "log_context"):
            self.log_context.__exit__(None, None, None)

        # 删除临时文件
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)

    def _create_test_data(self):
        """创建测试用的基础数据"""
        # 使用D.features读取数据
        self.instruments, self.start_time, self.end_time = read_corr_params("handler_config.yaml")
        base_fields = ["$close", "$open", "$high", "$low", "$volume"]

        df = D.features(self.instruments, base_fields, self.start_time, self.end_time)
        df.columns = [
            "close",
            "open",
            "high",
            "low",
            "volume",
        ]  # 重命名列以匹配后续处理

        # 确保数据类型为float64，以兼容TA-Lib
        for col in df.columns:
            df[col] = df[col].astype(np.float64)

        return df

    def test_read_corr_params(self):
        """测试read_corr_params函数"""
        # 测试正常情况
        instruments, start_time, end_time = read_corr_params(self.temp_config.name)

        # 验证返回值不为空
        self.assertIsNotNone(instruments)
        self.assertIsNotNone(start_time)
        self.assertIsNotNone(end_time)

        # 验证返回值的类型和内容
        self.assertIsInstance(instruments, list)
        self.assertEqual(len(instruments), 2)
        self.assertIn("000001.SZ", instruments)
        self.assertIn("000002.SZ", instruments)

        self.assertEqual(start_time, "2023-01-01")
        self.assertEqual(end_time, "2023-01-31")

        # 测试配置文件不存在的情况
        with self.assertRaises(Exception):
            read_corr_params("nonexistent_file.yaml")

    @patch("factors.factor_inspector.D")
    def test_factor_calculator_initialization(self, mock_d):
        """测试FactorCalculator的初始化"""
        calculator = FactorCalculator(self.test_expressions)
        # 验证表达式被正确存储
        self.assertEqual(calculator.expressions, self.test_expressions)

        # 测试无效的因子类型
        invalid_expressions = {"invalid_factor": {"type": "invalid_type"}}

        with self.assertRaises(ValueError):
            FactorCalculator(invalid_expressions)

    def test_talib_function_calculation(self):
        """测试TA-Lib函数因子计算"""
        calculator = FactorCalculator(self.test_expressions)

        # 测试SMA因子计算
        sma_result = calculator.calculate_factor("talib_factor", self.df_base)

        # 验证结果
        self.assertIsInstance(sma_result, pd.Series)
        self.assertEqual(len(sma_result), len(self.df_base))
        self.assertEqual(sma_result.index.equals(self.df_base.index), True)

        # 验证SMA计算结果的合理性
        # SMA应该是对close价格的5日移动平均
        expected_sma = self.df_base["close"].rolling(window=5).mean()

        # 由于TA-Lib和pandas的rolling可能有细微差异，我们检查前几个非NaN值
        valid_indices = ~sma_result.isna()
        if valid_indices.sum() > 0:
            # 检查前几个有效值的相对误差
            relative_error = (
                np.abs(sma_result[valid_indices] - expected_sma[valid_indices])
                / expected_sma[valid_indices]
            )
            self.assertTrue((relative_error < 0.01).all())  # 相对误差小于1%

    def test_python_code_calculation(self):
        """测试Python代码因子计算"""
        calculator = FactorCalculator(self.test_expressions)

        # 测试Python代码因子计算
        python_result = calculator.calculate_factor("python_factor", self.df_base)

        # 验证结果
        self.assertIsInstance(python_result, pd.Series)
        self.assertEqual(len(python_result), len(self.df_base))
        self.assertEqual(python_result.index.equals(self.df_base.index), True)

        # 验证计算结果
        expected_result = self.df_base["close"] / self.df_base["open"]
        pd.testing.assert_series_equal(
            python_result, expected_result, check_names=False
        )

    @patch("factors.factor_inspector.D")
    def test_qlib_expression_calculation(self, mock_d):
        """测试Qlib表达式因子计算"""
        # 模拟D.features的返回值
        mock_result = pd.DataFrame(
            {"close/open": self.df_base["close"] / self.df_base["open"]},
            index=self.df_base.index,
        )
        mock_d.features.return_value = mock_result

        calculator = FactorCalculator(self.test_expressions)

        # 测试Qlib表达式因子计算
        qlib_result = calculator.calculate_factor(
            "qlib_factor",
            self.df_base,
            instruments=self.instruments,
            start_time=self.start_time,
            end_time=self.end_time,
        )

        # 验证结果
        self.assertIsInstance(qlib_result, pd.Series)
        self.assertEqual(len(qlib_result), len(self.df_base))

        # 验证D.features被正确调用
        mock_d.features.assert_called_once()

    def test_calculate_all_factors(self):
        """测试批量计算所有因子"""
        calculator = FactorCalculator(self.test_expressions)

        # 批量计算所有因子
        results = calculator.calculate_all_factors(
            self.df_base, self.instruments, self.start_time, self.end_time
        )

        # 验证结果
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)  # 应该有3个因子

        # 验证每个因子都存在
        self.assertIn("qlib_factor", results)
        self.assertIn("talib_factor", results)
        self.assertIn("python_factor", results)

        # 验证每个因子的结果都是Series
        for factor_name, result in results.items():
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), len(self.df_base))
            self.assertEqual(result.index.equals(self.df_base.index), True)

    def build_factor_data(self, df_base):
        """
        使用TA-Lib函数生成因子数据作为基准

        Args:
            df_base (pd.DataFrame): 基础数据

        Returns:
            dict: 因子计算结果
        """
        try:
            import talib
        except ImportError:
            self.skipTest("TA-Lib not available")

        results = {}

        # 计算SMA因子
        sma_result = talib.SMA(df_base["close"].values, timeperiod=5)
        results["talib_factor"] = pd.Series(sma_result, index=df_base.index)

        # 计算Python代码因子
        python_result = df_base["close"] / df_base["open"]
        results["python_factor"] = python_result

        # 计算Qlib表达式因子（这里用简单的计算代替）
        qlib_result = df_base["close"] / df_base["open"]
        results["qlib_factor"] = qlib_result
        return results

    def test_factor_calculation_accuracy(self):
        """测试因子计算的准确性"""
        calculator = FactorCalculator(self.test_expressions)

        # 使用calculate_all_factors计算因子
        calculated_results = calculator.calculate_all_factors(
            self.df_base, self.instruments, self.start_time, self.end_time
        )

        # 使用build_factor_data生成基准数据
        expected_results = self.build_factor_data(self.df_base)

        # 比较结果
        for factor_name in self.test_expressions.keys():
            if factor_name in calculated_results and factor_name in expected_results:
                calculated = calculated_results[factor_name]
                expected = expected_results[factor_name]

                # 检查索引一致性
                self.assertTrue(calculated.index.equals(expected.index))

                # 对于非NaN的值，检查数值一致性
                valid_mask = ~(calculated.isna() | expected.isna())
                if valid_mask.sum() > 0:
                    # 计算相对误差
                    relative_error = np.abs(
                        calculated[valid_mask] - expected[valid_mask]
                    ) / np.abs(expected[valid_mask])
                    # 允许1%的相对误差
                    self.assertTrue(
                        (relative_error < 0.001).all(),
                        f"Factor {factor_name} has large relative error",
                    )

    def test_calculate_and_validate_factors(self):
        """测试calculate_and_validate_factors函数"""
        # 创建临时缓存目录
        with tempfile.TemporaryDirectory() as cache_dir:
            # 测试函数调用
            with patch("factors.factor_inspector.read_corr_params") as mock_read_params:
                mock_read_params.return_value = (
                    ["000001.SZ"],
                    "2023-01-01",
                    "2023-01-31",
                )

                with patch("factors.factor_inspector.D") as mock_d:
                    # 模拟D.features的返回值
                    mock_d.features.return_value = self.df_base

                    # 调用函数
                    calculate_and_validate_factors(self.test_expressions, cache_dir)

                    # 验证缓存文件被创建
                    factor_path = os.path.join(cache_dir, "factors.csv")
                    self.assertTrue(os.path.exists(factor_path))

                    # 验证生成的CSV文件内容
                    df_factors = pd.read_csv(factor_path, index_col=0)
                    self.assertGreater(len(df_factors.columns), 0)
                    self.assertGreater(len(df_factors), 0)


if __name__ == "__main__":
    unittest.main()
