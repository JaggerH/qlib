import argparse
import yaml
import importlib
import os


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
    args = parser.parse_args()

    handlers = load_handlers(args.config)
    factors = get_all_factors(handlers)

    if args.list:
        list_factors(factors)
    elif args.check:
        check_factor(factors, name=args.name, formula=args.formula)
    elif args.fuzzy:
        fuzzy_search(factors, args.fuzzy)
    else:
        parser.print_help()
