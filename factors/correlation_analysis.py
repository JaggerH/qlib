#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相关性分析脚本
读取handler.csv和factor.csv，分析factor列与handler中各列的相关性
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("开始进行相关性分析...")
    print("=" * 60)
    
    try:
        # 1. 读取CSV文件
        print("正在读取数据文件...")
        df_handler = pd.read_csv('factors/handler.csv')
        df_factor = pd.read_csv('factors/factor.csv')
        
        print(f"handler.csv 形状: {df_handler.shape}")
        print(f"factor.csv 形状: {df_factor.shape}")
        
        # 2. 设置MultiIndex
        print("\n正在设置MultiIndex...")
        df_handler = df_handler.set_index(['instrument', 'datetime'])
        df_factor = df_factor.set_index(['instrument', 'datetime'])
        
        # 3. 对齐数据，只保留交集
        print("正在对齐数据...")
        df_handler_aligned, df_factor_aligned = df_handler.align(df_factor, join='inner', axis=0)
        
        print(f"对齐后数据形状: {df_handler_aligned.shape}")
        print(f"共有 {len(df_handler_aligned)} 个有效数据点")
        
        # 4. 相关性分析
        print("\n" + "=" * 60)
        print("开始相关性分析...")
        print("=" * 60)
        
        correlations = {}
        factor_series = df_factor_aligned['factor']
        
        # 检查factor列是否有数据
        if factor_series.isnull().all():
            print("⚠️  警告: factor列全为空值，无法进行相关性分析")
            return
        
        valid_data_count = factor_series.notna().sum()
        print(f"factor列有效数据点: {valid_data_count}")
        
        # 计算每列的相关性
        high_correlation_count = 0
        for col in df_handler_aligned.columns:
            try:
                # 跳过全为NaN的列
                if df_handler_aligned[col].isnull().all():
                    continue
                
                # 计算相关性
                corr = df_handler_aligned[col].corr(factor_series)
                
                if not pd.isna(corr):
                    correlations[col] = corr
                    
            except Exception as e:
                print(f"计算 {col} 时出错: {e}")
                continue
        
        # 5. 输出结果
        print(f"\n成功计算了 {len(correlations)} 个因子的相关性")
        print("\n" + "=" * 60)
        print("相关性分析结果")
        print("=" * 60)
        
        # 按相关性绝对值排序
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for col, corr in sorted_correlations:
            abs_corr = abs(corr)
            if abs_corr > 0.5:
                print(f"🔥 【高相关性】{col:20s}: {corr:7.4f} (|{abs_corr:.4f}|)")
                high_correlation_count += 1
            elif abs_corr > 0.3:
                print(f"⚡ 【中等相关性】{col:20s}: {corr:7.4f} (|{abs_corr:.4f}|)")
            else:
                print(f"   {col:20s}: {corr:7.4f}")
        
        # 6. 总结
        print("\n" + "=" * 60)
        print("分析总结")
        print("=" * 60)
        print(f"总共分析因子数量: {len(correlations)}")
        print(f"高相关性因子数量 (|相关性| > 0.5): {high_correlation_count}")
        
        if high_correlation_count > 0:
            print(f"\n🚨 发现 {high_correlation_count} 个高相关性因子，请重点关注！")
            print("高相关性因子列表:")
            for col, corr in sorted_correlations:
                if abs(corr) > 0.5:
                    print(f"  - {col}: {corr:.4f}")
        else:
            print("\n✅ 未发现相关性大于0.5的因子")
            
        # 显示统计信息
        corr_values = list(correlations.values())
        print(f"\n相关性统计:")
        print(f"  最大相关性: {max(corr_values):.4f}")
        print(f"  最小相关性: {min(corr_values):.4f}")
        print(f"  平均相关性: {np.mean(corr_values):.4f}")
        print(f"  相关性标准差: {np.std(corr_values):.4f}")
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请确保 factors/handler.csv 和 factors/factor.csv 文件存在")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()