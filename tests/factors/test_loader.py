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
import os
import unittest
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from qlib.constant import REG_US
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


class TestCPyDL(unittest.TestCase):
    """CPyDL 详细测试用例"""

    def setUp(self):
        """测试前的准备工作"""
        # 创建临时测试配置文件
        self.test_factors = [
            {
                "name": "test_factor_1",
                "source": "custom_factor",
                "python_code": "result = close * 2",
                "inputs": ["$close"],
                "outputs": ["result"],
            },
            {
                "name": "test_factor_2",
                "source": "custom_factor",
                "python_code": "result = open + high",
                "inputs": ["$open", "$high"],
                "outputs": ["result"],
            },
            {
                "name": "qlib_factor",
                "source": "custom_factor",
                "qlib_expression": "Ref($close, -1)",
            },
        ]

        self.temp_dir = tempfile.mkdtemp()
        self.test_yaml_path = os.path.join(self.temp_dir, "test_factors.yaml")

        with open(self.test_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.test_factors, f, default_flow_style=False)

        # 测试数据
        self.test_instruments = ["000001.SZ", "000002.SZ"]
        self.test_start_time = "2020-01-01"
        self.test_end_time = "2020-01-10"

    def tearDown(self):
        """测试后的清理工作"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_init_with_yaml_path(self):
        """测试初始化时传入yaml_path"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)
        self.assertEqual(cpydl.yaml_path, self.test_yaml_path)
        self.assertEqual(len(cpydl.factor_configs), 2)  # 只有2个python_code因子

    def test_init_without_yaml_path(self):
        """测试初始化时不传入yaml_path（使用默认路径）"""
        # 这里需要mock默认路径下的factors.yaml
        with patch("os.path.join") as mock_join:
            mock_join.return_value = self.test_yaml_path
            cpydl = CPyDL()
            self.assertIsNotNone(cpydl.yaml_path)

    def test_filter_factors(self):
        """测试因子过滤逻辑"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)
        filtered_factors = cpydl._filter_factors(self.test_factors)

        # 应该只包含source为custom_factor且包含python_code的因子
        self.assertEqual(len(filtered_factors), 2)
        for factor in filtered_factors:
            self.assertEqual(factor["source"], "custom_factor")
            self.assertIn("python_code", factor)
            self.assertNotIn("type", factor)

    def test_compile_all_factors(self):
        """测试因子代码编译"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)

        # 验证编译后的因子
        self.assertIn("test_factor_1", cpydl.compiled_factors)
        self.assertIn("test_factor_2", cpydl.compiled_factors)

        # 验证编译的代码对象
        for factor_name, compiled_code in cpydl.compiled_factors.items():
            self.assertIsInstance(
                compiled_code, type(compile("pass", "<test>", "exec"))
            )

    def test_compile_factors_with_syntax_error(self):
        """测试编译包含语法错误的因子代码"""
        invalid_factors = [
            {
                "name": "invalid_factor",
                "source": "custom_factor",
                "python_code": "result = close * 2 +",  # 语法错误
                "inputs": ["$close"],
                "outputs": ["result"],
            }
        ]

        temp_yaml_path = os.path.join(self.temp_dir, "invalid_factors.yaml")
        with open(temp_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(invalid_factors, f, default_flow_style=False)

        with self.assertRaises(ValueError) as context:
            CPyDL(yaml_path=temp_yaml_path)
        self.assertIn("Failed to compile factor invalid_factor", str(context.exception))

    def test_build_dlwparser_config(self):
        """测试构建DLWParser配置"""
        import json
        cpydl = CPyDL(yaml_path=self.test_yaml_path)
        exprs, names = cpydl._build_dlwparser_config()

        # 验证表达式格式
        # expected_exprs = [
        #     json.dumps({"name": "test_factor_1", "result": "result"}),
        #     json.dumps({"name": "test_factor_2", "result": "result"})]
        # self.assertEqual(exprs, expected_exprs)

        # 验证名称
        expected_names = ["test_factor_1", "test_factor_2"]
        self.assertEqual(names, expected_names)

    def test_get_feature_config(self):
        """测试获取特征配置"""
        fields, names = CPyDL.get_feature_config(self.test_yaml_path)
        self.assertIsInstance(fields, list)
        self.assertIsInstance(names, list)
        self.assertEqual(len(fields), len(names))
        self.assertEqual(len(fields), 2)  # 2个python_code因子

    def test_collect_all_factor_inputs(self):
        """测试收集所有因子输入字段"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)
        inputs = cpydl._collect_all_factor_inputs()

        expected_inputs = ["$close", "$open", "$high"]
        self.assertEqual(set(inputs), set(expected_inputs))

    def test_get_factor_config_by_name(self):
        """测试根据名称获取因子配置"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)

        # 测试存在的因子
        factor_config = cpydl._get_factor_config_by_name("test_factor_1")
        self.assertEqual(factor_config["name"], "test_factor_1")
        self.assertEqual(factor_config["python_code"], "result = close * 2")

        # 测试不存在的因子
        with self.assertRaises(ValueError) as context:
            cpydl._get_factor_config_by_name("non_existent_factor")
        self.assertIn("Factor non_existent_factor not found", str(context.exception))

    def test_is_qlib_expression(self):
        """测试判断是否为qlib表达式"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)

        # 有效的qlib表达式
        self.assertTrue(cpydl._is_qlib_expression("Ref($close, -1)"))
        self.assertTrue(cpydl._is_qlib_expression("Mean($close, 5)"))

        # 无效的表达式
        self.assertFalse(cpydl._is_qlib_expression("invalid_expression"))
        self.assertFalse(cpydl._is_qlib_expression("test_factor_1_result"))

    def test_prepare_instruments(self):
        """测试准备股票代码列表"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)

        # 测试None值
        with patch("warnings.warn") as mock_warn:
            instruments = cpydl._prepare_instruments(None)
            mock_warn.assert_called_once()

        # 测试字符串值
        instruments = cpydl._prepare_instruments("all")
        self.assertIsNotNone(instruments)

        # 测试列表值
        instruments = cpydl._prepare_instruments(["000001.SZ", "000002.SZ"])
        self.assertEqual(instruments, ["000001.SZ", "000002.SZ"])

    def test_get_factor_name_from_expr(self):
        """测试从表达式中提取因子名称"""
        import json
        cpydl = CPyDL(yaml_path=self.test_yaml_path)

        # 标准格式
        factor_name = cpydl._get_factor_name_from_expr(json.dumps({"name": "test_factor_1", "result": "result"}), "result")
        self.assertEqual(factor_name, "test_factor_1")

        # 非标准格式
        factor_name = cpydl._get_factor_name_from_expr(json.dumps({"name": "test_factor_1", "result": "other"}), "other")
        self.assertEqual(factor_name, "test_factor_1")

    def test_get_qlib_expression_freq(self):
        """测试获取qlib表达式频率"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)
        freq = cpydl._get_qlib_expression_freq()
        self.assertEqual(freq, "day")

    @patch("qlib.data.D.features")
    def test_load_base_data(self, mock_features):
        """测试加载基础数据"""
        # 模拟返回的测试数据
        mock_df = pd.DataFrame(
            {
                "$close": [100, 101, 102],
                "$open": [99, 100, 101],
                "$high": [102, 103, 104],
            },
            index=pd.MultiIndex.from_arrays(
                [
                    ["000001.SZ", "000001.SZ", "000001.SZ"],
                    pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                ],
                names=["instrument", "datetime"],
            ),
        )

        mock_features.return_value = mock_df

        cpydl = CPyDL(yaml_path=self.test_yaml_path)
        base_data = cpydl._load_base_data(
            self.test_instruments, self.test_start_time, self.test_end_time
        )

        # 验证调用
        mock_features.assert_called_once()
        call_args = mock_features.call_args
        self.assertEqual(call_args[0][0], self.test_instruments)
        self.assertEqual(set(call_args[0][1]), set(["$close", "$open", "$high"]))
        self.assertEqual(call_args[0][2], self.test_start_time)
        self.assertEqual(call_args[0][3], self.test_end_time)

        # 验证返回数据
        self.assertIsInstance(base_data, pd.DataFrame)

    def test_load_base_data_empty_inputs(self):
        """测试加载基础数据时输入字段为空"""
        # 创建没有输入字段的因子配置
        empty_input_factors = [
            {
                "name": "no_input_factor",
                "source": "custom_factor",
                "python_code": "result = 1.0",
                "inputs": [],
                "outputs": ["result"],
            }
        ]

        temp_yaml_path = os.path.join(self.temp_dir, "empty_input_factors.yaml")
        with open(temp_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(empty_input_factors, f, default_flow_style=False)

        cpydl = CPyDL(yaml_path=temp_yaml_path)
        base_data = cpydl._load_base_data(
            self.test_instruments, self.test_start_time, self.test_end_time
        )

        self.assertTrue(base_data.empty)

    def test_prepare_execution_environment(self):
        """测试准备执行环境"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)
        # inputs = cpydl._collect_all_factor_inputs()
        inputs = ["$close", "$open", "$high"]

        # 创建测试输入数据
        test_data = pd.DataFrame(
            {
                "$close": [100, 101, 102],
                "$open": [99, 100, 101],
                "$high": [102, 103, 104],
                "custom_field": [1, 2, 3],
            }
        )

        exec_env = cpydl._prepare_execution_environment(test_data, inputs)

        # 验证映射的字段
        self.assertIn("close", exec_env)
        self.assertIn("open", exec_env)
        self.assertIn("high", exec_env)
        self.assertNotIn("custom_field", exec_env)

        # 验证库导入
        self.assertIn("pd", exec_env)
        self.assertIn("np", exec_env)

    def test_validate_factor_output(self):
        """测试验证因子输出"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)

        # 创建有效的Series
        valid_series = pd.Series(
            [1, 2, 3],
            index=pd.MultiIndex.from_arrays(
                [
                    ["000001.SZ", "000001.SZ", "000001.SZ"],
                    pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                ],
                names=["instrument", "datetime"],
            ),
        )

        # 应该不抛出异常
        cpydl._validate_factor_output(valid_series, "test_factor")

        # 测试无效的Series（错误的索引名称）
        invalid_series = pd.Series(
            [1, 2, 3],
            index=pd.MultiIndex.from_arrays(
                [
                    ["000001.SZ", "000001.SZ", "000001.SZ"],
                    pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                ],
                names=["stock", "date"],
            ),  # 错误的索引名称
        )

        with self.assertRaises(ValueError) as context:
            cpydl._validate_factor_output(invalid_series, "test_factor")
        self.assertIn(
            "must have MultiIndex with names ['datetime', 'instrument']",
            str(context.exception),
        )

        # 测试非Series/DataFrame类型
        with self.assertRaises(ValueError) as context:
            cpydl._validate_factor_output([1, 2, 3], "test_factor")
        self.assertIn(
            "must return a pandas.Series or pandas.DataFrame", str(context.exception)
        )

    def test_build_factor_results_df(self):
        """测试构建因子结果DataFrame"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)

        # 创建测试结果
        test_results = [
            pd.Series(
                [1, 2, 3],
                index=pd.MultiIndex.from_arrays(
                    [
                        ["000001.SZ", "000001.SZ", "000001.SZ"],
                        pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                    ],
                    names=["instrument", "datetime"],
                ),
            ),
            pd.Series(
                [4, 5, 6],
                index=pd.MultiIndex.from_arrays(
                    [
                        ["000001.SZ", "000001.SZ", "000001.SZ"],
                        pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                    ],
                    names=["instrument", "datetime"],
                ),
            ),
        ]
        names = ["factor1", "factor2"]

        result_df = cpydl._build_factor_results_df(test_results, names)

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df.columns), 2)
        self.assertIn("factor1", result_df.columns)
        self.assertIn("factor2", result_df.columns)

    def test_build_factor_results_df_empty(self):
        """测试构建空的因子结果DataFrame"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)

        result_df = cpydl._build_factor_results_df([], [])

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertTrue(result_df.empty)
        self.assertEqual(result_df.index.names, ["instrument", "datetime"])

    @patch("qlib.data.D.features")
    def test_calculate_qlib_expression(self, mock_features):
        """测试计算qlib表达式"""
        # 模拟返回数据
        mock_df = pd.DataFrame(
            {"Ref($close, -1)": [99, 100, 101]},
            index=pd.MultiIndex.from_arrays(
                [
                    ["000001.SZ", "000001.SZ", "000001.SZ"],
                    pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                ],
                names=["instrument", "datetime"],
            ),
        )

        mock_features.return_value = mock_df

        cpydl = CPyDL(yaml_path=self.test_yaml_path)
        result = cpydl._calculate_qlib_expression(
            "Ref($close, -1)",
            self.test_instruments,
            self.test_start_time,
            self.test_end_time,
        )

        # 验证调用
        mock_features.assert_called_once()
        call_args = mock_features.call_args
        self.assertEqual(call_args[0][0], self.test_instruments)
        self.assertEqual(call_args[0][1], ["Ref($close, -1)"])
        self.assertEqual(call_args[0][2], self.test_start_time)
        self.assertEqual(call_args[0][3], self.test_end_time)

        # 验证返回结果
        self.assertIsInstance(result, pd.Series)

    def test_execute_factor_with_base_data(self):
        """测试使用基础数据执行因子计算"""
        cpydl = CPyDL(yaml_path=self.test_yaml_path)

        # 创建测试基础数据
        base_data = pd.DataFrame(
            {
                "$close": [100, 101, 102],
                "$open": [99, 100, 101],
                "$high": [102, 103, 104],
            },
            index=pd.MultiIndex.from_arrays(
                [
                    ["000001.SZ", "000001.SZ", "000001.SZ"],
                    pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                ],
                names=["instrument", "datetime"],
            ),
        )

        factor_config = {
            "name": "test_factor_1",
            "python_code": "result = close * 2",
            "inputs": ["$close"],
            "outputs": ["result"],
        }

        results = cpydl._execute_factor_with_base_data(factor_config, base_data)

        self.assertIn("result", results)
        self.assertIsInstance(results["result"], pd.Series)
        # 验证计算结果
        self.assertEqual(results["result"].iloc[0], 200)  # 100 * 2

    def test_load_group_df_integration(self):
        """测试load_group_df集成功能"""
        # 这个测试需要完整的数据环境，可能需要mock
        pass

    def test_load_group_df_with_qlib_expression(self):
        """测试load_group_df处理qlib表达式"""
        # 这个测试需要完整的数据环境，可能需要mock
        pass

    def test_load_group_df_with_custom_factor(self):
        """测试load_group_df处理自定义因子"""
        # 这个测试需要完整的数据环境，可能需要mock
        pass


class TestCPyDLIntegration(unittest.TestCase):
    """CPyDL 集成测试"""

    def test_CPyDL_full_pipeline(self):
        """测试CPyDL完整流程"""
        fields, names = CPyDL.get_feature_config()
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


class TestCIntradayDL(unittest.TestCase):
    """CIntradayDL 集成测试"""

    def test_CIntradayDL_full_pipeline(self):
        """测试CIntradayDL完整流程"""
        fields, names = CIntradayDL.get_feature_config()
        self.assertIsInstance(names, list)

        instruments, start_time, end_time = read_corr_params()

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
