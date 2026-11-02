#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理脚本：从独立的 forecasts.pkl 和 tss.pkl 文件加载预测结果，
计算评估指标、绘制 PIT 直方图和预测样例图，并保存到同一目录。

用法：
    python evaluate.py --dir /path/to/exp_dir
或：
    python evaluate.py --forecast_file ./log/T/2026-03-17-16-34-23/forecasts.pkl --tss_file ./log/T/2026-03-17-16-34-23/tss.pkl
python evaluate.py --forecast_file ./log_compar/T/2026-03-18-09-58-38/forecasts.pkl --tss_file ./log_compar/T/2026-03-18-09-58-38/tss.pkl
python evaluate.py --forecast_file ./log_compar2/2026-03-18-11-48-11/forecasts.pkl --tss_file ./log_compar2/2026-03-18-11-48-11/tss.pkl
python evaluate.py --forecast_file ./log_compar3/T/2026-03-18-16-11-51/forecasts.pkl --tss_file ./log_compar3/T/2026-03-18-16-11-51/tss.pkl


    
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.evaluation import Evaluator
import CRPS.CRPS as pscore

def set_spine_width(ax, width=2):
    """设置坐标轴线宽"""
    for spine in ax.spines.values():
        spine.set_linewidth(width)


def load_forecasts_and_tss(forecast_path, tss_path):
    """加载两个 pickle 文件"""
    with open(forecast_path, 'rb') as f:
        forecasts = pickle.load(f)
    with open(tss_path, 'rb') as f:
        tss = pickle.load(f)
    return forecasts, tss





def plot_sample_forecasts(forecasts, tss, prediction_length, save_dir, max_samples=21):
    """绘制前 max_samples 个预测样例图并保存"""
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 12})

    for idx, (forecast, ts) in enumerate(zip(forecasts, tss)):
        if idx >= max_samples:
            break

        fig, ax = plt.subplots(figsize=(8, 6))

        # 获取真实值的最后 3 倍预测长度
        true_values = ts[-5 * prediction_length:].values
        length = len(true_values)
        x_axis = np.arange(1, length + 1)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.plot(x_axis, true_values, 'k-', label='True Values', linewidth=2)

        # 预测部分
        pred_len = prediction_length
        pred_x = np.arange(length - pred_len + 1, length + 1)

        lower_90 = forecast.quantile(0.20)
        upper_90 = forecast.quantile(0.80)
        lower_50 = forecast.quantile(0.40)
        upper_50 = forecast.quantile(0.60)

        ax.fill_between(pred_x, lower_90, upper_90, color='blue', alpha=0.4, label='90% Interval')
        ax.fill_between(pred_x, lower_50, upper_50, color='blue', alpha=0.2, label='50% Interval')

        set_spine_width(ax, width=2)
        ax.tick_params(axis='both', labelsize=10, width=1.5, length=6)

        ax.set_xlim(left=-3, right=length + 3)
        max_tick = ((length - 1) // 30) * 30
        xticks = np.arange(0, max_tick + 31, 30)
        xticks = xticks[xticks <= length]
        ax.set_xticks(xticks)

        ax.set_title('Probabilistic Forecast of TFT on Exchange', fontsize=14)
        plt.tight_layout()
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)

        out_path = os.path.join(save_dir, f'forecast_{idx}.pdf')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='从保存的 forecasts.pkl 和 tss.pkl 生成评估图表')
    parser.add_argument('--dir', type=str, help='存放 forecasts.pkl 和 tss.pkl 的目录')
    parser.add_argument('--forecast_file', type=str, help='forecasts.pkl 的完整路径')
    parser.add_argument('--tss_file', type=str, help='tss.pkl 的完整路径')
    args = parser.parse_args()

    # 确定文件路径
    if args.dir:
        base_dir = args.dir
        forecast_path = os.path.join(base_dir, 'forecasts.pkl')
        tss_path = os.path.join(base_dir, 'tss.pkl')
    elif args.forecast_file and args.tss_file:
        forecast_path = args.forecast_file
        tss_path = args.tss_file
        base_dir = os.path.dirname(forecast_path)  # 假设两个文件在同一目录
    else:
        parser.error("请提供 --dir 或同时提供 --forecast_file 和 --tss_file")

    # 加载数据
    print(f"加载 forecasts: {forecast_path}")
    print(f"加载 tss: {tss_path}")
    forecasts, tss = load_forecasts_and_tss(forecast_path, tss_path)
    print(f"成功加载 {len(forecasts)} 个预测序列")

    # 获取预测长度（假设一致）
    prediction_length = forecasts[0].prediction_length
    print(f"预测长度: {prediction_length}")


    # 创建 plots 子目录
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)


    # 绘制预测样例图
    print("绘制前21个预测样例图...")
    plot_sample_forecasts(forecasts, tss, prediction_length, plots_dir, max_samples=21)
    print(f"预测样例图已保存至: {plots_dir}")


if __name__ == "__main__":
    main()