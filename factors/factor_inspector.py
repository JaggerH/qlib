import argparse
import yaml
import importlib
import os
import pandas as pd
from qlib.data import D
import qlib
from qlib.config import REG_CN


def load_handlers(config_path):
    # 保证 config_path 是和当前脚本同级目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, config_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    handlers = []
    for item in config["handlers"]:
        module = importlib.import_module(item["module"])
        handler_cls = getattr(module, item["class"])
        handlers.append(handler_cls)
    return handlers


def get_all_factors(handlers):
    all_factors = {}
    for handler in handlers:
        fields, names = handler.get_feature_config()
        for name, field in zip(names, fields):
            all_factors[name] = field
    return all_factors


def list_factors(factors):
    for name, formula in factors.items():
        print(f"{name}: {formula}")


def check_factor(factors, name=None, formula=None):
    for n, f in factors.items():
        if name and n == name:
            print(f"因子名已存在: {n}")
            return True
        if formula and f == formula:
            print(f"因子公式已存在: {f}")
            return True
    print("未发现重复，可以添加新因子。")
    return False


def fuzzy_search(factors, keyword):
    found = False
    for n, f in factors.items():
        if keyword.lower() in n.lower() or keyword.lower() in f.lower():
            print(f"{n}: {f}")
            found = True
    if not found:
        print(f"未找到包含 '{keyword}' 的因子名或公式。")


def convert_datetime_index(df):
    # 获取 instrument 和 datetime 两个 level
    instruments = df.index.get_level_values("instrument")
    datetimes = pd.to_datetime(
        df.index.get_level_values("datetime")
    )  # 转换为 datetime64[ns]
    # 重新构造 MultiIndex
    df.index = pd.MultiIndex.from_arrays([instruments, datetimes], names=df.index.names)
    return df


def read_corr_params(config_path):
    try:
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


def check_and_generate_handler_csv(
    handler_csv_path, factors, instruments, start_time, end_time
):
    try:
        handler_df = pd.read_csv(handler_csv_path, index_col=["instrument", "datetime"])
        if len(handler_df.columns) != len(factors):
            raise ValueError(
                "handler.csv列数与factors数量不一致，重新生成handler.csv..."
            )
        return convert_datetime_index(handler_df)
    except Exception as e:
        qlib.init(region=REG_CN)
        df = D.features(instruments, list(factors.values()), start_time, end_time)
        df.columns = factors.keys()
        df.to_csv(handler_csv_path)
        return df


def compare_factor_corr(factors, factor_csv_path, factor_col="factor", threshold=0.5):
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    handler_csv_path = os.path.join(cache_dir, "handler.csv")

    # 读取参数
    instruments, start_time, end_time = read_corr_params("handler_config.yaml")
    handler_df = check_and_generate_handler_csv(
        handler_csv_path, factors, instruments, start_time, end_time
    )

    factor_df = pd.read_csv(factor_csv_path, index_col=["instrument", "datetime"])
    factor_df = convert_datetime_index(factor_df)
    handler_df, factor_df = handler_df.align(factor_df, join="inner", axis=0)

    corrs = []
    for col in handler_df.columns:
        if handler_df[col].isnull().all():
            continue
        corr = handler_df[col].corr(factor_df[factor_col])
        corrs.append((col, corr))
    # 按相关性绝对值从大到小排序
    corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
    # 输出前十
    for col, corr in corrs_sorted[:10]:
        if abs(corr) > threshold:
            print(f"【重点】{col}: 相关性={corr:.3f}")
        else:
            print(f"{col}: 相关性={corr:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="因子注册表工具")
    parser.add_argument(
        "--config",
        type=str,
        default="handler_config.yaml",
        help="DataHandler 配置文件路径（始终在本脚本同级目录下查找）",
    )
    parser.add_argument("--list", action="store_true", help="列出所有因子")
    parser.add_argument("--check", action="store_true", help="检查因子是否存在")
    parser.add_argument("--name", type=str, help="要检查的因子名")
    parser.add_argument("--formula", type=str, help="要检查的因子公式")
    parser.add_argument("--fuzzy", type=str, help="模糊查找因子名或公式，支持部分匹配")
    parser.add_argument(
        "--corr", type=str, help="指定新因子csv，计算与所有handler因子的相关性"
    )

    args = parser.parse_args()

    handlers = load_handlers(args.config)
    factors = get_all_factors(handlers)

    if args.list:
        list_factors(factors)
    elif args.check:
        check_factor(factors, name=args.name, formula=args.formula)
    elif args.fuzzy:
        fuzzy_search(factors, args.fuzzy)
    elif args.corr:
        compare_factor_corr(factors, args.corr)
    else:
        parser.print_help()
