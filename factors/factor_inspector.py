import os
import yaml
import argparse

import pandas as pd
import qlib
from qlib.data import D
from qlib.config import REG_CN
from factors.handler import CombineHandler, TestFactorHandler

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)


def read_corr_params(config_path=None):
    """
    读取handler_config.yaml中的参数
    """
    try:
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "handler_config.yaml"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        corr_params = config.get("corr_params", {})
        required_fields = ["instruments", "start_time", "end_time"]
        for field in required_fields:
            if field not in corr_params:
                raise ValueError(f"缺少必需字段: {field}")
        instruments = corr_params["instruments"]
        start_time = corr_params["start_time"]
        end_time = corr_params["end_time"]
        return instruments, start_time, end_time
    except Exception as e:
        raise e


def get_handler_config(config_path=None):
    instruments, start_time, end_time = read_corr_params(config_path)
    return {
        "start_time": start_time,
        "end_time": end_time,
        "instruments": instruments,
    }


def get_all_factors(config_path=None):
    config = get_handler_config(config_path)
    handler = CombineHandler(**config)
    return handler.get_feature_config()[1]  # return names


def convert_datetime_index(df):
    """
    将handler.csv的index从['instrument', 'datetime']转换为['instrument', 'datetime']
    """
    # 获取 instrument 和 datetime 两个 level
    instruments = df.index.get_level_values("instrument")
    datetimes = pd.to_datetime(
        df.index.get_level_values("datetime")
    )  # 转换为 datetime64[ns]
    # 重新构造 MultiIndex
    df.index = pd.MultiIndex.from_arrays([instruments, datetimes], names=df.index.names)
    return df


def generate_handler_csv(handler_csv_path="cache/handler.csv"):
    try:
        factors = get_all_factors()
        handler_df = pd.read_csv(handler_csv_path, index_col=["instrument", "datetime"])
        if len(handler_df.columns) != len(factors):
            raise ValueError(
                "handler.csv列数与factors数量不一致，重新生成handler.csv..."
            )
        return convert_datetime_index(handler_df)
    except Exception as e:
        print(f"重新生成handler.csv: {e}")
        # 使用get_mixed_handler_data函数获取数据
        config = get_handler_config()
        handler = CombineHandler(**config)
        df = handler.fetch(col_set="feature")
        df.to_csv(handler_csv_path)


def generate_factor_csv(handler_yaml_path, factor_csv_path="cache/factor.csv"):
    config = get_handler_config()
    config["yaml_path"] = handler_yaml_path
    handler = TestFactorHandler(**config)
    df = handler.fetch(col_set="feature")
    df.to_csv(factor_csv_path)


def compare_factor_corr_from_files(
    handler_csv_path="cache/handler.csv",
    factor_csv_path="cache/factor.csv",
    threshold=0.5,
):
    """
    从CSV文件读取数据并进行因子相关性分析

    Args:
        handler_csv_path (str): handler数据文件路径
        factor_csv_path (str): 待分析因子数据文件路径
        threshold (float): 相关性阈值
    """
    try:
        handler_df = pd.read_csv(handler_csv_path, index_col=[0, 1], parse_dates=True)

        factor_df = pd.read_csv(factor_csv_path, index_col=[0, 1], parse_dates=True)

        # 对齐数据
        handler_df, factor_df = handler_df.align(factor_df, join="inner", axis=0)

        # 对每个输入的因子进行分析
        for input_factor in factor_df.columns:
            print(f"\n分析因子: {input_factor}")
            print("-" * 50)

            corrs = []
            for handler_factor in handler_df.columns:
                # 安全地检查是否为全空值
                try:
                    if handler_df[handler_factor].isnull().all():
                        continue
                except ValueError:
                    # 如果出现ValueError，跳过这个因子
                    continue
                corr = handler_df[handler_factor].corr(factor_df[input_factor])
                if not pd.isna(corr):
                    corrs.append((handler_factor, corr))

            # 按相关性绝对值从大到小排序
            corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
            # 输出前十
            for col, corr in corrs_sorted[:5]:
                if abs(corr) > threshold:
                    print(f"【重点】{col}: 相关性={corr:.3f}")
                else:
                    print(f"{col}: 相关性={corr:.3f}")

    except Exception as e:
        print(f"相关性分析出错: {e}")


def main():
    """
    主函数，处理命令行参数并执行因子相关性分析流程
    """
    parser = argparse.ArgumentParser(description="因子相关性分析工具")
    parser.add_argument(
        "--factor_yaml", type=str, required=True, help="指定因子参数的yaml文件路径"
    )
    parser.add_argument(
        "--handler_csv",
        type=str,
        default="cache/handler.csv",
        help="handler数据文件路径 (默认: cache/handler.csv)",
    )
    parser.add_argument(
        "--factor_csv",
        type=str,
        default="cache/factor.csv",
        help="因子数据文件路径 (默认: cache/factor.csv)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="相关性阈值 (默认: 0.5)"
    )

    args = parser.parse_args()

    try:
        print(f"因子YAML文件: {args.factor_yaml}")
        generate_handler_csv(args.handler_csv)
        generate_factor_csv(args.factor_yaml, args.factor_csv)
        compare_factor_corr_from_files(
            args.handler_csv, args.factor_csv, args.threshold
        )
        print("✓ 因子相关性分析完成")
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
