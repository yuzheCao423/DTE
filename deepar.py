import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from loggers_comparison import *
import json 
from gluonts.dataset.repository import get_dataset, dataset_names
from gluonts.dataset.util import to_pandas
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import CRPS.CRPS as pscore
import os
import random
import argparse
import lightning.pytorch as pl
import torch
import pickle
from itertools import islice

def run_single_experiment(seed=None):
    """
    执行一次完整的实验流程（训练+评估）。
    如果指定seed，则设置随机种子以保证可重复性（不同seed产生不同结果）。
    返回包含各项指标的字典。
    """
    if seed is not None:
        pl.seed_everything(seed, workers=True)
        # 额外设置其他库的随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", type=str, required=True)
    argparser.add_argument("-id", "--expid", type=int, required=True)
    args = argparser.parse_args()
    
    config = json.load(open(args.config,'r'))
    logger = Logger()
    exp_agent = Experiment_Agent(config=config, id=args.expid)
    # exp_agent.init_new_exp()
    exp_agent.run_exp(timestamp = logger.timestamp)

    dataset_name = config["dataset"]
    context_length = config["context_length"]
    prediction_length = config.get("predict_length", None)
    print("prediction_length:", prediction_length)
    # %%
    dataset = get_dataset(config["dataset"])

    
    estimator = DeepAREstimator(freq = dataset.metadata.freq, prediction_length=config["predict_length"],context_length=config["context_length"],
                            trainer_kwargs = {"max_epochs":config["epochs"]}, batch_size=config["batch_size"])

    # %%
    t_start = datetime.now()
    predictor = estimator.train(dataset.train)
    t_end = datetime.now()

    # %%
    logger.finish(config=config)


    # %%
    e_start = datetime.now()
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,  # test dataset
        predictor=predictor,  # predictor
        num_samples=1000,  # number of sample paths we want for evaluation
    )
    e_end = datetime.now()

    # %%
    forecasts_pytorch = list(forecast_it)
    tss_pytorch = list(ts_it)
    # forecasts_pytorch = list(f.to_sample_forecast() for f in forecast_it)
    # tss_pytorch = list(ts_it)

    forecast_save_path = os.path.join(logger.dirname, 'forecasts.pkl')
    tss_save_path = os.path.join(logger.dirname, 'tss.pkl')
    with open(forecast_save_path, 'wb') as f:
        pickle.dump(forecasts_pytorch, f)
    with open(tss_save_path, 'wb') as f:
        pickle.dump(tss_pytorch, f)
    print(f"Forecasts saved to {forecast_save_path}")
    print(f"Time series saved to {tss_save_path}")

    # ---------- 新增：计算自定义评估指标和PIT直方图 ----------
    pit_values = []               # 存储所有PIT值
    picp_count = 0                # 落在90%区间内的点数
    total_points = 0               # 总预测点数
    total_width = 0.0              # 区间宽度总和
    global_min = float('inf')
    global_max = -float('inf')
    crps_values = []              

    # 先收集所有真实值的全局范围
    for forecast, ts in zip(forecasts_pytorch, tss_pytorch):
        start = forecast.start_date
        end = start + forecast.prediction_length * forecast.freq
        true_slice = ts[start:end].values
        if len(true_slice) != forecast.prediction_length:
            continue
        global_min = min(global_min, true_slice.min())
        global_max = max(global_max, true_slice.max())

    print('evaluate...')
    # 遍历每个测试序列
    for forecast, ts in zip(forecasts_pytorch, tss_pytorch):
        start = forecast.start_date
        end = start + forecast.prediction_length * forecast.freq
        true_slice = ts[start:end].values
        if len(true_slice) != forecast.prediction_length:
            continue

        samples = forecast.samples                     # (num_samples, pred_len)
        num_samples, pred_len = samples.shape

        # 计算0.05和0.95分位数
        lower_90 = np.percentile(samples, 5, axis=0)
        upper_90 = np.percentile(samples, 95, axis=0)

        # PICP统计
        covered = (true_slice.reshape(-1) >= lower_90) & (true_slice.reshape(-1) <= upper_90)
        picp_count += covered.sum()
        total_points += pred_len

        # 宽度统计
        total_width += (upper_90 - lower_90).sum()

        # PIT值计算
        for t in range(pred_len):
            y = true_slice[t]
            sample_t = samples[:, t]
            pit = (np.sum(sample_t < y) + 0.5 * np.sum(sample_t == y)) / num_samples
            pit_values.append(pit)

        # for t in range(pred_len):
        #     y = true_slice[t]
        #     sample_t = samples[:, t]  # 所有样本在该时间点的值
        #     crps,_,_ = pscore(sample_t.reshape(-1,1), y).compute()/(global_max - global_min)        
            
        #     crps_values.append(crps)
        # print(len(crps_values))

    # 计算最终指标
    picp = picp_count / total_points if total_points > 0 else np.nan
    avg_width = total_width / total_points if total_points > 0 else np.nan
    pinaw = avg_width / (global_max - global_min) if (global_max - global_min) > 0 else np.nan

    # 使用GluonTS Evaluator获得标准指标
    evaluator = Evaluator(quantiles=(np.arange(20)/20.0)[1:])
    metrics_pytorch, _ = evaluator(tss_pytorch, forecasts_pytorch)

    # 合并指标字典
    metrics_dict = dict(metrics_pytorch)   # 确保是可变字典
    metrics_dict['PICP_0.9'] = picp
    metrics_dict['PINAW'] = pinaw
    # crps_mean=0
    # for i in range(len(crps_values)):
    #     crps_mean+=crps_values[i]
    # crps_mean=crps_mean/(len(crps_values))
    # metrics_dict['CRPS_manual'] = crps_mean

    # 绘制PIT直方图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(pit_values, bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.axhline(y=1, color='red', linestyle='--', label='Uniform')
    ax.set_xlabel('PIT')
    ax.set_ylabel('Density')
    ax.set_title('Probability Integral Transform (PIT) Histogram')
    ax.legend()
    plt.tight_layout()
    pit_plot_path = os.path.join(logger.plot_dirname, 'pit_histogram.pdf')
    plt.savefig(pit_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 保存合并后的指标到CSV
    df_metrics = pd.DataFrame([metrics_dict])
    import csv
    path = logger.evaluate_to_log_name()
    # 如果失败，可以这样写：
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(df_metrics.columns.tolist())  # 列名
        writer.writerows(df_metrics.values.tolist())   # 数据行
    
    # ---------- 预测结果可视化 ----------
    def set_spine_width(ax, width=2):
        for spine in ax.spines.values():
            spine.set_linewidth(width)    
    # 设置全局字体大小
    plt.rcParams.update({"font.size": 12})    
    # 遍历前21个测试样本
    for idx, (forecast, ts) in islice(enumerate(zip(forecasts_pytorch, tss_pytorch)), 21):
        # 创建新图形
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 获取真实值的最后3倍预测长度
        true_values = ts[-5 * prediction_length:].values
        length = len(true_values)
        x_axis = np.arange(1, length + 1)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        
        # 绘制真实值
        ax.plot(x_axis, true_values, 'k-', label='True Values', linewidth=2)
        
        # 预测部分
        pred_len = prediction_length
        pred_x = np.arange(length - pred_len + 1, length + 1)
        
        # 获取分位数并确保为一维
        median = forecast.quantile(0.5)
        lower_90 = forecast.quantile(0.05)
        upper_90 = forecast.quantile(0.95)
        lower_50 = forecast.quantile(0.25)
        upper_50 = forecast.quantile(0.75)
        
        
        # 绘制90%置信区间（alpha=0.4, 蓝色）
        ax.fill_between(pred_x, lower_90, upper_90, color='blue', alpha=0.4, label='90% Interval')
        # 绘制50%置信区间（alpha=0.2, 蓝色）
        ax.fill_between(pred_x, lower_50, upper_50, color='blue', alpha=0.2, label='50% Interval')
        
        # 边框加粗
        set_spine_width(ax, width=2)
        # 设置刻度字体大小
        ax.tick_params(axis='both', labelsize=10, width=1.5, length=6)
        
        # 设置x轴范围：左侧从0开始，右侧稍微超出数据点
        ax.set_xlim(left=-3, right=length + 3)  # 右侧留一点边距
        # 横坐标刻度：从0开始，每隔50显示一个，直到不超过数据长度
        max_tick = ((length - 1) // 30) * 30   
        xticks = np.arange(0, max_tick + 31, 30)  
        xticks = xticks[xticks <= length]         # 只保留 ≤ length 的刻度
        ax.set_xticks(xticks)
        
        # 设置标题字体大小
        ax.set_title('Probabilistic Forecast of DeepAR', fontsize=14)
        
        plt.gcf().tight_layout()
        # 设置图例字体大小
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        output_path = os.path.join(logger.plot_dirname, f'forecast_{idx}.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    return metrics_dict


def main():
    """运行5次实验，汇总指标并保存统计结果"""
    n_runs = 3
    seeds = list(range(n_runs))  # 使用不同的随机种子
    all_metrics = []

    for i, seed in enumerate(seeds):
        print(f"\n========== Running experiment {i+1}/{n_runs} with seed {seed} ==========")
        metrics = run_single_experiment(seed=seed)
        all_metrics.append(metrics)
        # 清理GPU缓存（如果使用）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 将所有指标转换为DataFrame
    df_all = pd.DataFrame(all_metrics)
    # 计算平均值和标准差
    # summary = df_all.agg(['mean', 'std']).T  # 转置使每个指标一行，列名为mean和std
    summary = df_all.apply(lambda col: col.apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x)).agg(['mean', 'std']).T
    summary.columns = ['mean', 'std']
    summary.to_csv('results/deepar_exchange_2T.csv', index=True)
    print("\nSummary statistics saved to summary_results.csv")
    print(summary)


if __name__ == "__main__":
    main()


    


