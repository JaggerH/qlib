import argparse
import yaml
import importlib
import os
import pandas as pd
from qlib.data import D
import qlib
from qlib.config import REG_CN
from typing import Dict, List, Optional, Any
import json
import time
import re

# 尝试加载python-dotenv来支持.env文件
try:
    from dotenv import load_dotenv

    # 自动加载.env文件
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print(
        "Warning: python-dotenv not available. Install with: pip install python-dotenv"
    )
    print("You can still use system environment variables.")


def replace_env_vars(config_str: str) -> str:
    """替换配置字符串中的环境变量

    支持格式: ${VAR_NAME} 或 $VAR_NAME
    """

    def replace_func(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))  # 如果环境变量不存在，保持原样

    # 替换 ${VAR_NAME} 格式
    config_str = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace_func, config_str)
    # 替换 $VAR_NAME 格式
    config_str = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", replace_func, config_str)

    return config_str


def load_config_with_env_vars(config_path: str) -> Dict[str, Any]:
    """加载配置文件并替换环境变量"""
    with open(config_path, "r", encoding="utf-8") as f:
        config_str = f.read()

    # 替换环境变量
    config_str = replace_env_vars(config_str)

    # 解析YAML
    return yaml.safe_load(config_str)


# 设置代理环境变量（需要在导入 google.generativeai 之前设置）
def setup_proxy_if_configured():
    """从配置文件设置代理"""
    try:
        config = load_config_with_env_vars("handler_config.yaml")

        embedding_config = config.get("embedding_config", {})
        proxy_url = embedding_config.get("proxy_url")

        if proxy_url:
            os.environ["http_proxy"] = proxy_url
            os.environ["https_proxy"] = proxy_url
            print(f"Proxy configured: {proxy_url}")
            return True
    except Exception as e:
        print(f"Warning: Could not load proxy config: {e}")
    return False


# 在导入 Gemini API 之前设置代理
setup_proxy_if_configured()

# Gemini API 导入
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print(
        "Warning: google-generativeai not available. Install with: pip install google-generativeai"
    )

# 其他 Embedding 相关导入
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(
        "Warning: sentence-transformers not available. Install with: pip install sentence-transformers"
    )

try:
    import faiss
    import numpy as np

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available. Install with: pip install faiss-cpu")


def load_embedding_config(config_path: str) -> Dict[str, Any]:
    """从配置文件加载Embedding相关配置"""
    config = load_config_with_env_vars(config_path)
    return config.get("embedding_config", {})


def generate_factor_text_for_embedding(factor: Dict[str, Any]) -> str:
    """为因子生成用于embedding的文本"""
    parts = [
        f"因子名称: {factor.get('name', '')}",
        f"Qlib表达式: {factor.get('qlib_expression', '')}",
        f"来源: {factor.get('source', '')}",
    ]

    # 添加其他可用字段
    if "formula" in factor:
        parts.append(f"数学公式: {factor['formula']}")
    if "description" in factor:
        parts.append(f"描述: {factor['description']}")
    if "tags" in factor and factor["tags"]:
        parts.append(f"标签: {', '.join(factor['tags'])}")

    return "; ".join(parts)


def generate_embedding_gemini(text: str, config: Dict[str, Any]) -> List[float]:
    """使用Gemini API生成embedding"""
    if not GEMINI_AVAILABLE:
        print("Warning: Gemini API not available")
        return []

    api_key = config.get("api_key")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Warning: Please set a valid Gemini API key in handler_config.yaml")
        return []

    max_retries = config.get("max_retries", 3)
    timeout = config.get("timeout", 30)

    for attempt in range(max_retries):
        try:
            genai.configure(api_key=api_key)
            model = config.get("model", "models/embedding-001")

            # 生成embedding
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document",
                request_options={"timeout": timeout},
            )

            return result["embedding"]

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 递增等待时间
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(
                    f"Warning: Failed to generate Gemini embedding after {max_retries} attempts: {e}"
                )
                return []

    return []


def generate_embedding_local(text: str, config: Dict[str, Any]) -> List[float]:
    """使用本地模型生成embedding"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Warning: sentence-transformers not available")
        return []

    try:
        model_name = config.get("local_model", "paraphrase-multilingual-MiniLM-L12-v2")
        model = SentenceTransformer(model_name)
        embedding = model.encode(text).tolist()
        return embedding
    except Exception as e:
        print(f"Warning: Failed to generate local embedding: {e}")
        return []


def generate_embedding(text: str, config: Dict[str, Any]) -> List[float]:
    """根据配置生成embedding"""
    provider = config.get("provider", "gemini")

    if provider == "gemini":
        return generate_embedding_gemini(text, config)
    elif provider == "local":
        return generate_embedding_local(text, config)
    else:
        print(f"Warning: Unknown embedding provider: {provider}")
        return []


def batch_generate_embeddings_for_factors(
    factors: List[Dict],
    config: Dict[str, Any],
    existing_embeddings: Dict[str, List[float]] = None,
) -> List[Dict]:
    """批量为因子生成embedding（增量模式）"""
    if existing_embeddings is None:
        existing_embeddings = {}

    # 统计需要生成embedding的因子
    factors_need_embedding = []
    factors_with_embedding = []

    for factor in factors:
        factor_name = factor.get("name")
        # 检查是否已有embedding（在内存、已存储的embedding文件中）
        has_embedding = False

        if "embedding" in factor and factor["embedding"]:
            # 内存中已有embedding
            has_embedding = True
            factors_with_embedding.append(factor)
        elif factor_name and factor_name in existing_embeddings:
            # 从存储中加载embedding
            factor_copy = factor.copy()
            factor_copy["embedding"] = existing_embeddings[factor_name]
            factors_with_embedding.append(factor_copy)
            has_embedding = True

        if not has_embedding:
            factors_need_embedding.append(factor)

    print(f"Total factors: {len(factors)}")
    print(f"Factors with existing embedding: {len(factors_with_embedding)}")
    print(f"Factors needing new embedding: {len(factors_need_embedding)}")

    if not factors_need_embedding:
        print("All factors already have embeddings!")
        return factors_with_embedding

    print(f"\nGenerating embeddings for {len(factors_need_embedding)} new factors...")

    batch_size = config.get("batch_size", 10)
    request_delay = config.get("request_delay", 1)

    updated_factors = factors_with_embedding.copy()

    for i, factor in enumerate(factors_need_embedding):
        factor_name = factor.get("name", "Unknown")
        print(
            f"Generating embedding for factor {i+1}/{len(factors_need_embedding)}: {factor_name}"
        )

        # 生成embedding文本
        embedding_text = generate_factor_text_for_embedding(factor)

        # 生成embedding
        embedding = generate_embedding(embedding_text, config)

        if embedding:
            factor_copy = factor.copy()
            factor_copy["embedding"] = embedding
            updated_factors.append(factor_copy)
            print(f"Successfully generated embedding for {factor_name}")
        else:
            print(f"Failed to generate embedding for {factor_name}")
            updated_factors.append(factor)

        # 添加延时
        if i < len(factors_need_embedding) - 1:  # 最后一个不需要延时
            time.sleep(request_delay)

    return updated_factors


def build_faiss_index(factors: List[Dict]) -> tuple:
    """构建FAISS索引"""
    if not FAISS_AVAILABLE:
        print("Warning: FAISS not available")
        return None, []

    # 收集有embedding的因子
    factors_with_embedding = []
    embeddings = []

    for factor in factors:
        if "embedding" in factor and factor["embedding"]:
            factors_with_embedding.append(factor)
            embeddings.append(factor["embedding"])

    if not embeddings:
        print("No factors with embeddings found")
        return None, []

    # 构建FAISS索引
    embeddings_array = np.array(embeddings).astype("float32")
    dimension = embeddings_array.shape[1]

    # 使用L2距离的索引
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    print(f"Built FAISS index with {len(factors_with_embedding)} factors")
    return index, factors_with_embedding


def search_similar_factors(
    query_text: str,
    index,
    factors_with_embedding: List[Dict],
    config: Dict[str, Any],
    top_k: int = 5,
) -> List[tuple]:
    """搜索相似因子"""
    if not index or not factors_with_embedding:
        print("No FAISS index or factors available")
        return []

    # 生成查询embedding
    query_embedding = generate_embedding(query_text, config)
    if not query_embedding:
        print("Failed to generate query embedding")
        return []

    # 搜索
    query_vector = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(
        query_vector, min(top_k, len(factors_with_embedding))
    )

    # 返回结果
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(factors_with_embedding):
            similarity = 1 / (1 + distance)  # 转换为相似度分数
            results.append((factors_with_embedding[idx], similarity, distance))

    return results


def load_factors_yaml(yaml_path: str, load_embeddings: bool = True) -> List[Dict]:
    """加载factors.yaml文件，可选择是否加载embedding数据"""
    import base64
    import json

    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            factors = yaml.safe_load(f) or []

        # 如果需要加载embedding
        if load_embeddings:
            # 先检查是否有分离的embedding文件
            embedding_path = yaml_path.replace(".yaml", "_embeddings.yaml")
            if os.path.exists(embedding_path):
                embeddings = load_embeddings_yaml(embedding_path)
                factors = merge_factors_with_embeddings(factors, embeddings)
            else:
                # 兼容旧格式（base64和原始数据）
                for factor in factors:
                    # 处理base64编码的embedding
                    if "_embedding_b64" in factor:
                        try:
                            embedding_json = base64.b64decode(
                                factor["_embedding_b64"]
                            ).decode("utf-8")
                            factor["embedding"] = json.loads(embedding_json)
                        except Exception as e:
                            print(
                                f"Warning: Failed to decode embedding for {factor.get('name', 'Unknown')}: {e}"
                            )
                            factor["embedding"] = []

                    # 兼容更旧格式
                    elif "_embedding_data" in factor:
                        factor["embedding"] = factor["_embedding_data"]

        return factors
    return []


def save_factors_yaml(factors: List[Dict], yaml_path: str):
    """保存因子列表到yaml文件，将embedding分离存储"""
    # 创建干净的因子副本（不包含embedding数据）
    clean_factors = []
    embeddings_data = {}

    for factor in factors:
        factor_copy = factor.copy()

        # 如果有embedding数据，分离存储
        if "embedding" in factor_copy and factor_copy["embedding"]:
            factor_name = factor_copy.get("name")
            if factor_name:
                embeddings_data[factor_name] = factor_copy["embedding"]
            # 从主文件中移除embedding相关字段
            factor_copy.pop("embedding", None)
            factor_copy.pop("_embedding_b64", None)
            factor_copy.pop("_embedding_data", None)

        clean_factors.append(factor_copy)

    # 保存主factors文件（无embedding）
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(clean_factors, f, allow_unicode=True, sort_keys=False)

    # 保存embedding文件（如果有embedding数据）
    if embeddings_data:
        embedding_path = yaml_path.replace(".yaml", "_embeddings.yaml")
        save_embeddings_yaml(embeddings_data, embedding_path)


def save_embeddings_yaml(embeddings_data: Dict[str, List[float]], embedding_path: str):
    """保存embedding数据到独立的yaml文件"""
    with open(embedding_path, "w", encoding="utf-8") as f:
        yaml.dump(embeddings_data, f, allow_unicode=True, sort_keys=False)
    print(f"Embeddings saved to: {embedding_path}")


def load_embeddings_yaml(embedding_path: str) -> Dict[str, List[float]]:
    """加载embedding数据"""
    if os.path.exists(embedding_path):
        with open(embedding_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def merge_factors_with_embeddings(
    factors: List[Dict], embeddings: Dict[str, List[float]]
) -> List[Dict]:
    """将因子数据与embedding数据合并"""
    merged_factors = []
    for factor in factors:
        factor_copy = factor.copy()
        factor_name = factor_copy.get("name")
        if factor_name and factor_name in embeddings:
            factor_copy["embedding"] = embeddings[factor_name]
        merged_factors.append(factor_copy)
    return merged_factors


def export_handlers_to_yaml(handlers: List, yaml_path: str = "factors/factors.yaml"):
    """将所有handler中的因子导出到yaml文件

    Args:
        handlers: handler类列表
        yaml_path: yaml文件路径
    """
    # 加载现有的factors.yaml
    existing_factors = load_factors_yaml(yaml_path)
    existing_names = {f["name"] for f in existing_factors}

    # 获取handler中的所有因子
    new_factors = []
    for handler in handlers:
        fields, names = handler.get_feature_config()
        # 根据handler类名判断来源
        source = handler.__name__

        for name, expression in zip(names, fields):
            # 跳过已存在的因子
            if name in existing_names:
                continue

            # 只包含基本信息
            factor = {"name": name, "qlib_expression": expression, "source": source}
            new_factors.append(factor)

    # 合并新旧因子列表
    all_factors = existing_factors + new_factors

    # 保存到yaml文件
    save_factors_yaml(all_factors, yaml_path)
    print(f"Added {len(new_factors)} new factors to {yaml_path}")
    return new_factors


def load_handlers(config_path):
    # 保证 config_path 是和当前脚本同级目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, config_path)
    config = load_config_with_env_vars(yaml_path)
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
        config = load_config_with_env_vars(config_path)
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


def compare_factor_corr(factors, factor_csv_path, threshold=0.5):
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

    # 对每个输入的因子进行分析
    for input_factor in factor_df.columns:
        print(f"\n分析因子: {input_factor}")
        print("-" * 50)

        corrs = []
        for handler_factor in handler_df.columns:
            if handler_df[handler_factor].isnull().all():
                continue
            corr = handler_df[handler_factor].corr(factor_df[input_factor])
            corrs.append((handler_factor, corr))

        # 按相关性绝对值从大到小排序
        corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
        # 输出前十
        for col, corr in corrs_sorted[:10]:
            if abs(corr) > threshold:
                print(f"【重点】{col}: 相关性={corr:.3f}")
            else:
                print(f"{col}: 相关性={corr:.3f}")


def calculate_and_validate_factors(expressions, cache_dir="cache"):
    """
    计算因子值并进行验证。

    Args:
        expressions (dict): 因子名称和表达式的字典
        cache_dir (str): 缓存目录路径，默认为"cache"
    """
    # 读取配置参数
    instruments, start_time, end_time = read_corr_params("handler_config.yaml")

    # 初始化qlib
    qlib.init(region=REG_CN)

    # 获取因子数据
    df = D.features(instruments, list(expressions.values()), start_time, end_time)
    df.columns = expressions.keys()

    # 保存因子数据
    os.makedirs(cache_dir, exist_ok=True)
    factor_path = os.path.join(cache_dir, "factors.csv")
    df.to_csv(factor_path)

    # 运行因子检验
    compare_factor_corr(
        get_all_factors(load_handlers("handler_config.yaml")), factor_path
    )


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
    parser.add_argument(
        "--export-yaml",
        type=str,
        help="将所有handler中的因子导出到指定的yaml文件",
    )
    parser.add_argument(
        "--generate-embeddings",
        type=str,
        help="为指定yaml文件中的因子生成embedding向量",
    )
    parser.add_argument("--search-similar", type=str, help="搜索与给定文本相似的因子")
    parser.add_argument("--factors-file", type=str, help="指定因子文件路径（可选）")
    parser.add_argument(
        "--view-factors",
        type=str,
        help="查看指定yaml文件中的因子信息（不显示embedding）",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="不加载embedding数据（加快文件加载速度）",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="返回最相似的k个因子（默认5个）"
    )

    args = parser.parse_args()

    if args.generate_embeddings:
        # 生成embedding（增量模式）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(script_dir, args.config)
        embedding_config = load_embedding_config(config_file_path)

        # 加载因子文件（不加载embedding，让增量逻辑处理）
        factors = load_factors_yaml(args.generate_embeddings, load_embeddings=False)
        if factors:
            # 检查是否有已存在的embedding文件
            embedding_path = args.generate_embeddings.replace(
                ".yaml", "_embeddings.yaml"
            )
            existing_embeddings = {}
            if os.path.exists(embedding_path):
                print(f"Loading existing embeddings from: {embedding_path}")
                existing_embeddings = load_embeddings_yaml(embedding_path)

            # 增量生成embedding
            updated_factors = batch_generate_embeddings_for_factors(
                factors, embedding_config, existing_embeddings
            )

            # 保存更新后的因子和embedding
            save_factors_yaml(updated_factors, args.generate_embeddings)
            print(f"Updated embeddings saved to {args.generate_embeddings}")
        else:
            print(f"No factors found in {args.generate_embeddings}")

    elif args.search_similar:
        # 搜索相似因子
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(script_dir, args.config)
        embedding_config = load_embedding_config(config_file_path)

        # 确定因子文件路径
        if args.factors_file:
            factors_file = args.factors_file
        else:
            # 如果在factors目录下运行，使用当前目录的factors.yaml
            if os.path.basename(os.getcwd()) == "factors":
                factors_file = "factors.yaml"
            else:
                factors_file = "factors/factors.yaml"

        factors = load_factors_yaml(factors_file)

        if factors:
            index, factors_with_embedding = build_faiss_index(factors)
            if index:
                results = search_similar_factors(
                    args.search_similar,
                    index,
                    factors_with_embedding,
                    embedding_config,
                    args.top_k,
                )

                if results:
                    print(f"\n找到 {len(results)} 个相似因子:")
                    print("-" * 60)
                    for i, (factor, similarity, distance) in enumerate(results, 1):
                        print(
                            f"{i}. {factor.get('name', 'Unknown')} (相似度: {similarity:.3f})"
                        )
                        print(f"   表达式: {factor.get('qlib_expression', 'N/A')}")
                        print(f"   来源: {factor.get('source', 'N/A')}")
                        if "formula" in factor:
                            print(f"   公式: {factor['formula']}")
                        if "tags" in factor:
                            print(f"   标签: {', '.join(factor['tags'])}")
                        print()
                else:
                    print("未找到相似因子")
            else:
                print("无法构建FAISS索引")
        else:
            print(f"无法加载因子文件: {factors_file}")

    elif args.view_factors:
        # 查看因子信息（不显示embedding）
        load_embeddings = not args.no_embeddings
        factors = load_factors_yaml(args.view_factors, load_embeddings=load_embeddings)
        if factors:
            print(f"\n因子文件: {args.view_factors}")
            print(f"总共包含 {len(factors)} 个因子\n")
            print("=" * 80)

            for i, factor in enumerate(factors, 1):
                print(f"{i}. {factor.get('name', 'Unknown')}")
                print(f"   表达式: {factor.get('qlib_expression', 'N/A')}")
                print(f"   来源: {factor.get('source', 'N/A')}")

                if "description" in factor:
                    print(f"   描述: {factor['description']}")
                if "formula" in factor:
                    print(f"   公式: {factor['formula']}")
                if "tags" in factor and factor["tags"]:
                    print(f"   标签: {', '.join(factor['tags'])}")
                if "metadata" in factor and factor["metadata"]:
                    print(f"   元数据: {', '.join(factor['metadata'])}")

                # 显示embedding维度而不是具体值
                if "embedding" in factor and factor["embedding"]:
                    print(f"   Embedding: {len(factor['embedding'])} 维向量")
                else:
                    # 检查是否有独立的embedding文件
                    embedding_path = args.view_factors.replace(
                        ".yaml", "_embeddings.yaml"
                    )
                    if os.path.exists(embedding_path):
                        embeddings = load_embeddings_yaml(embedding_path)
                        if factor.get("name") in embeddings:
                            print(
                                f"   Embedding: {len(embeddings[factor.get('name')])} 维向量 (分离存储)"
                            )
                        else:
                            print("   Embedding: 未生成")
                    else:
                        print("   Embedding: 未生成")

                print("-" * 80)
        else:
            print(f"无法加载因子文件: {args.view_factors}")

    else:
        # 原有功能
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
        elif args.export_yaml:
            export_handlers_to_yaml(handlers, args.export_yaml)
        else:
            parser.print_help()
