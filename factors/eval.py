import warnings
import pandas as pd
import numpy as np

import seaborn as sns
import math
import copy

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-bright")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def factor_ic_filter(
    dataset,
    label="LABEL0",
    rank_ic_threshold=0.01,
    corr_threshold=0.9,
):
    """
    批量计算 Rank IC 并筛选因子，同时去掉高度相关的冗余因子

    参数:
    - dataset: Qlib Dataset 对象，必须先 prepare("train") 得到 DataFrame
    - label_col: 目标列，默认 "LABEL0"
    - rank_ic_threshold: Rank IC 绝对值阈值，低于该值的因子会被过滤掉
    - corr_threshold: 因子间 Pearson 相关系数阈值，超过该值的冗余因子会被过滤掉

    返回:
    - selected_factors: 最终筛选后的因子列表（只含因子名）
    - ic_df: Rank IC DataFrame
    - corr_matrix: 筛选后因子相关系数矩阵
    """
    # 1. 获取 train 数据
    df = dataset.prepare("train", col_set=["feature", "label"])

    # 2. 分离因子列和 label 列
    factor_cols = [col for col in df.columns if col[0] == "feature"]
    label_col = ("label", label)
    y = df[label_col]

    # 3. 计算 Rank IC
    rank_ic_dict = {}
    for col in factor_cols:
        factor_series = df[col]
        mask = factor_series.notna() & y.notna()
        if mask.sum() == 0:
            rank_ic_dict[col[1]] = np.nan
            continue

        rank_factor = factor_series[mask].rank()
        rank_label = y[mask].rank()
        corr = rank_factor.corr(rank_label)
        rank_ic_dict[col[1]] = corr

    ic_df = pd.DataFrame({"Rank_IC": pd.Series(rank_ic_dict)})

    # 4. 初步筛选 Rank IC
    ic_selected = ic_df[ic_df["Rank_IC"].abs() > rank_ic_threshold].index.tolist()

    if not ic_selected:
        return [], ic_df, None

    # 5. 筛选后的因子 DataFrame
    df_selected = df[[("feature", f) for f in ic_selected]].copy()

    # 6. 计算因子间相关性矩阵
    corr_matrix = df_selected.corr()

    # 7. 去掉高度相关的冗余因子
    # 贪心法：按 Rank IC 从大到小排序，保留高 IC 因子，去掉与已选因子 corr > corr_threshold 的因子
    sorted_factors = (
        ic_df.loc[ic_selected].sort_values("Rank_IC", ascending=False).index.tolist()
    )
    final_factors = []

    for f in sorted_factors:
        keep = True
        for kept in final_factors:
            if abs(corr_matrix[("feature", f)][("feature", kept)]) > corr_threshold:
                keep = False
                break
        if keep:
            final_factors.append(f)

    # 8. 添加 warning
    if len(final_factors) < len(df) / 15:
        warnings.warn(
            f"筛选后的因子数量 {len(final_factors)} 低于样本数量/15 ({len(df)//15})，可能过少"
        )

    # 9. 返回
    return final_factors  # , ic_df, corr_matrix


def update_dataset_config(dataset_config, factors):
    """
    更新数据集配置，添加因子筛选处理器

    参数:
    - dataset_config: Qlib dataset config
    - factors: 要保留的因子列表

    返回:
    - dataset_config: 更新后的数据集对象，在 learn_processors 和 infer_processors 中添加了 FilterCol 处理器
    """
    _dataset_config = copy.deepcopy(dataset_config)
    _dataset_config["kwargs"]["handler"]["kwargs"]["learn_processors"].insert(
        0, {"class": "FilterCol", "kwargs": {"col_list": factors}}
    )
    _dataset_config["kwargs"]["handler"]["kwargs"]["infer_processors"].insert(
        0, {"class": "FilterCol", "kwargs": {"col_list": factors}}
    )
    return _dataset_config


def plot_ic_headmap(dataset):
    """
    绘制因子与标签的相关性热力图

    将因子按组显示（每组最多10个因子），生成多个子图的热力图，
    展示因子与标签（LABEL0）之间的相关性关系

    参数:
    - dataset: Qlib Dataset 对象，必须先 prepare("train") 得到 DataFrame

    返回:
    - None: 直接显示热力图，不返回值
    """
    df = dataset.prepare("train", col_set=["feature", "label"])
    corr = df.corr()

    # 获取所有因子列（去除label相关列）
    all_columns = [col for col in corr.columns if col not in ["label", "LABEL0"]]
    n_factors = len(all_columns)
    group_size = 10
    n_groups = math.ceil(n_factors / group_size)

    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    for i in range(n_groups):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        start = i * group_size
        end = min((i + 1) * group_size, n_factors)
        factor_cols = all_columns[start:end]
        # 每个热力图都包含label和LABEL0
        cols = [("label", "LABEL0")] + factor_cols
        # 只保留这些列和行
        sub_corr = corr.loc[cols, cols]
        sns.heatmap(sub_corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"因子相关性: {start}-{end-1}")

    # 去除多余的子图
    for j in range(n_groups, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()


def plot_segment(dataset, valid_date, test_date):
    """
    绘制数据集划分的可视化图表

    显示训练集、验证集、测试集的标签（LABEL0）时间序列，
    并在图上标注划分点位置

    参数:
    - dataset: Qlib Dataset 对象，必须包含 train、valid、test 三个数据集
    - valid_date: 验证集开始日期，用于绘制垂直分割线
    - test_date: 测试集开始日期，用于绘制垂直分割线

    返回:
    - None: 直接显示图表，不返回值
    """
    train_data = dataset.prepare("train")
    valid_data = dataset.prepare("valid")
    test_data = dataset.prepare("test")

    plt.figure(figsize=(15, 6))
    plt.plot(
        train_data.index.get_level_values("datetime"),
        train_data["LABEL0"],
        label="训练集",
        color="blue",
    )
    plt.plot(
        valid_data.index.get_level_values("datetime"),
        valid_data["LABEL0"],
        label="验证集",
        color="green",
    )
    plt.plot(
        test_data.index.get_level_values("datetime"),
        test_data["LABEL0"],
        label="测试集",
        color="red",
    )
    plt.axvline(valid_date, color="black", linestyle="--", label="划分点")
    plt.axvline(test_date, color="black", linestyle="--", label="划分点")
    plt.title("训练集、验证集、测试集划分")
    plt.xlabel("日期")
    plt.ylabel("收益率")
    plt.legend()
    plt.grid(True)
    plt.show()
