"""
可视化不同时延下 Stackelberg博弈 vs 传统方法的对比结果
"""

import json
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
from glob import glob
import matplotlib.patheffects as path_effects
from matplotlib.legend_handler import HandlerTuple
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
import matplotlib.colors as colors
from tidewave_usbl import TideWave, USBL

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 指标中文名称映射
METRIC_NAMES_CN = {
    "episode_reward": "回合奖励",
    "ep_reward": "回合奖励",
    "avg_detJ": "平均Fisher信息矩阵detJ",
    "max_detJ": "最大Fisher信息矩阵detJ",
    "sum_rate": "总数据率",
    "data_collection_rate": "数据收集率",
    "idu": "数据更新次数",
    "Ec": "平均能耗",
    "energy_consumption": "平均能耗",
    "N_DO": "数据溢出率",
    "data_overflows": "数据溢出率",
    "avg_tracking_error": "平均跟踪误差",
    "tracking_error": "跟踪误差",
    "avg_usv_move": "USV平均移动距离",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--result_file",
    type=str,
    default=None,
    help="延迟对比结果文件路径（JSON或Pickle），如果不指定则使用最新的结果",
)
parser.add_argument(
    "--output_dir", type=str, default="delay_comparison_results", help="输出目录"
)

# 只在直接运行时解析参数，作为模块导入时不解析
if __name__ == "__main__":
    args = parser.parse_args()
else:
    # 作为模块导入时，创建一个默认的args对象
    class DefaultArgs:
        result_file = None
        output_dir = "delay_comparison_results"
    args = DefaultArgs()


def load_latest_result():
    """加载最新的延迟对比结果（支持子文件夹）"""
    result_dir = args.output_dir
    # 先查找子文件夹中的文件
    subdirs = glob(f"{result_dir}/delay_comparison_*/")
    if subdirs:
        # 找到最新的子文件夹
        latest_dir = max(subdirs, key=os.path.getctime)
        json_files = glob(f"{latest_dir}delay_comparison_*.json")
        if json_files:
            return max(json_files, key=os.path.getctime)
    # 如果没有子文件夹，查找根目录下的文件（向后兼容）
    json_files = glob(f"{result_dir}/delay_comparison_*.json")
    if json_files:
        latest_file = max(json_files, key=os.path.getctime)
        return latest_file
    return None


def load_delay_comparison_data(file_path):
    """加载延迟对比数据"""
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)
        # 尝试加载对应的pickle文件以获取完整数据
        pickle_path = file_path.replace(".json", ".pkl")
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                full_data = pickle.load(f)
                return full_data
        return data
    elif file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("不支持的文件格式")


def get_delay_title_text(delay_value):
    """获取延迟标题文本"""
    # delay_value实际上是丢包模式标志(0或1)
    is_packet_loss = (delay_value > 0.5)
    
    formula = r"$Delay = T_{trans} + T_{fixed} + T_{sample}$"
    loss_text = "有丢包" if is_packet_loss else "无丢包"
    
    return f"{formula} ({loss_text})"


def plot_detJ_evolution_for_delay(trad_results, stack_results, delay, save_path, timestamp):
    """绘制detJ值随时间的变化"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    title_text = get_delay_title_text(delay)
    
    # 处理传统方法
    if trad_results and len(trad_results) > 0 and "detJ_values" in trad_results[0]:
        trad_detJ_all = []
        for result in trad_results:
            if "detJ_values" in result and "detJ_timestamps" in result:
                trad_detJ_all.append((result["detJ_timestamps"], result["detJ_values"]))
        
        if trad_detJ_all:
            max_len = max(len(ts) for ts, _ in trad_detJ_all)
            trad_detJ_avg = np.zeros(max_len)
            trad_detJ_std = np.zeros(max_len)
            count = np.zeros(max_len)
            
            for ts, values in trad_detJ_all:
                for i, (t, v) in enumerate(zip(ts, values)):
                    if i < max_len:
                        trad_detJ_avg[i] += v
                        trad_detJ_std[i] += v**2
                        count[i] += 1
            
            trad_detJ_avg /= count
            trad_detJ_std = np.sqrt(np.maximum(0, trad_detJ_std / count - trad_detJ_avg**2))
            trad_timestamps = np.arange(max_len)
            
            ax.plot(trad_timestamps, trad_detJ_avg, label="传统方法", color="#3498db", linewidth=2)
            ax.fill_between(trad_timestamps, trad_detJ_avg - trad_detJ_std, 
                          trad_detJ_avg + trad_detJ_std, alpha=0.3, color="#3498db")
    
    # 处理Stackelberg方法
    if stack_results and len(stack_results) > 0 and "detJ_values" in stack_results[0]:
        stack_detJ_all = []
        for result in stack_results:
            if "detJ_values" in result and "detJ_timestamps" in result:
                stack_detJ_all.append((result["detJ_timestamps"], result["detJ_values"]))
        
        if stack_detJ_all:
            max_len = max(len(ts) for ts, _ in stack_detJ_all)
            stack_detJ_avg = np.zeros(max_len)
            stack_detJ_std = np.zeros(max_len)
            count = np.zeros(max_len)
            
            for ts, values in stack_detJ_all:
                for i, (t, v) in enumerate(zip(ts, values)):
                    if i < max_len:
                        stack_detJ_avg[i] += v
                        stack_detJ_std[i] += v**2
                        count[i] += 1
            
            stack_detJ_avg /= count
            stack_detJ_std = np.sqrt(np.maximum(0, stack_detJ_std / count - stack_detJ_avg**2))
            stack_timestamps = np.arange(max_len)
            
            ax.plot(stack_timestamps, stack_detJ_avg, label="Stackelberg方法", 
                   color="#e74c3c", linewidth=2)
            ax.fill_between(stack_timestamps, stack_detJ_avg - stack_detJ_std, 
                          stack_detJ_avg + stack_detJ_std, alpha=0.3, color="#e74c3c")
    
    ax.set_xlabel("时间步")
    ax.set_ylabel("Fisher信息矩阵 detJ值")
    ax.set_title(f"{title_text}\nFisher信息矩阵detJ值随时间变化对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{save_path}/detJ_evolution_delay_{delay}_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ detJ演化图: {output_path}")
    plt.close()


def plot_summary_table_for_delay(trad_stats, stack_stats, delay, save_path, timestamp, 
                                  trad_results=None, stack_results=None):
    """为单个时延场景生成对比总结表格图"""
    title_text = get_delay_title_text(delay)
    
    # 基础指标
    base_metrics = ["avg_detJ", "sum_rate", "Ec"]
    
    table_data = []
    
    for metric in base_metrics:
        if metric in trad_stats and metric in stack_stats:
            # 获取数据
            trad_val = trad_stats[metric]
            stack_val = stack_stats[metric]
            
            # 兼容处理
            if isinstance(trad_val, dict) and "mean" in trad_val:
                trad_mean = trad_val["mean"]
            else:
                trad_mean = float(trad_val)
                
            if isinstance(stack_val, dict) and "mean" in stack_val:
                stack_mean = stack_val["mean"]
            else:
                stack_mean = float(stack_val)
            
            # 计算改进百分比
            if trad_mean != 0:
                improvement = (stack_mean - trad_mean) / abs(trad_mean) * 100
            else:
                improvement = 0
            
            # 获取中文名称
            if metric == "avg_detJ": metric_name = "平均Fisher信息矩阵"
            elif metric == "sum_rate": metric_name = "总数据率"
            elif metric == "Ec": metric_name = "平均能耗"
            else: metric_name = metric
            
            # 颜色标记逻辑 (基于理论预期)
            if metric == "avg_detJ":
                is_good = improvement < 0  # 预期减小
            elif metric == "sum_rate":
                is_good = improvement > 0  # 预期增大
            elif metric == "Ec":
                is_good = improvement > 0  # 预期增大
            else:
                is_good = improvement > 0
            
            # 格式化
            if metric == "avg_detJ":
                trad_str = f"{trad_mean:.4e}"
                stack_str = f"{stack_mean:.4e}"
            else:
                trad_str = f"{trad_mean:.4f}"
                stack_str = f"{stack_mean:.4f}"
            
            table_data.append([
                metric_name, trad_str, stack_str, f"{improvement:+.2f}%", is_good
            ])
    
    # 从results中计算每个AUV的追踪误差
    if trad_results and stack_results:
        # 确定AUV数量
        n_auv = 0
        for r in trad_results:
            if "avg_tracking_error" in r and isinstance(r["avg_tracking_error"], list):
                n_auv = len(r["avg_tracking_error"])
                break
        
        if n_auv > 0:
            # 为每个AUV计算平均追踪误差
            for auv_idx in range(n_auv):
                trad_errors = []
                stack_errors = []
                
                for r in trad_results:
                    if "avg_tracking_error" in r and isinstance(r["avg_tracking_error"], list):
                        if auv_idx < len(r["avg_tracking_error"]):
                            trad_errors.append(r["avg_tracking_error"][auv_idx])
                
                for r in stack_results:
                    if "avg_tracking_error" in r and isinstance(r["avg_tracking_error"], list):
                        if auv_idx < len(r["avg_tracking_error"]):
                            stack_errors.append(r["avg_tracking_error"][auv_idx])
                
                if trad_errors and stack_errors:
                    trad_mean = np.mean(trad_errors)
                    stack_mean = np.mean(stack_errors)
                    
                    if trad_mean != 0:
                        imp = (stack_mean - trad_mean) / abs(trad_mean) * 100
                    else:
                        imp = 0
                    is_good = imp < 0
                    
                    table_data.append([
                        f"AUV{auv_idx} 追踪误差", f"{trad_mean:.4f}", f"{stack_mean:.4f}", f"{imp:+.2f}%", is_good
                    ])
    else:
        # 回退到原来的逻辑（使用stats中的聚合数据）
        if "avg_tracking_error" in trad_stats and "avg_tracking_error" in stack_stats:
            trad_val = trad_stats["avg_tracking_error"]
            stack_val = stack_stats["avg_tracking_error"]
            
            # 兼容列表或字典
            if isinstance(trad_val, dict) and "mean" in trad_val:
                trad_mean = trad_val["mean"]
            elif isinstance(trad_val, list):
                trad_mean = np.mean(trad_val)
            else:
                trad_mean = float(trad_val)
                
            if isinstance(stack_val, dict) and "mean" in stack_val:
                stack_mean = stack_val["mean"]
            elif isinstance(stack_val, list):
                stack_mean = np.mean(stack_val)
            else:
                stack_mean = float(stack_val)
                
            if trad_mean != 0:
                imp = (stack_mean - trad_mean) / abs(trad_mean) * 100
            else:
                imp = 0
            is_good = imp < 0
            table_data.append([
                "平均追踪误差", f"{trad_mean:.4f}", f"{stack_mean:.4f}", f"{imp:+.2f}%", is_good
            ])
    
    if not table_data:
        return
    
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.8 + 1))
    ax.axis("tight")
    ax.axis("off")
    
    display_data = [row[:4] for row in table_data]
    
    table = ax.table(cellText=display_data,
                    colLabels=["指标", "传统方法", "Stackelberg方法", "改进"],
                    cellLoc="center", loc="center",
                    colWidths=[0.3, 0.2, 0.2, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(4):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    for i in range(1, len(table_data) + 1):
        is_good = table_data[i - 1][4]
        if is_good:
            table[(i, 3)].set_facecolor("#d5f4e6")
        else:
            table[(i, 3)].set_facecolor("#ffe5e5")
    
    plt.title(f"{title_text}\nStackelberg博弈 vs 传统方法对比总结", 
             fontsize=14, fontweight="bold", pad=20)
    
    output_path = f"{save_path}/comparison_summary_delay_{delay}_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ 对比总结表: {output_path}")
    plt.close()


def plot_detJ_evolution_for_delay_three_columns(trad_rt_results, trad_del_results, stack_results, delay, save_path, timestamp):
    """绘制detJ值随时间的变化（三列对比）"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    title_text = get_delay_title_text(delay)
    
    # 处理实时传统方法
    if trad_rt_results and len(trad_rt_results) > 0 and "detJ_values" in trad_rt_results[0]:
        trad_rt_detJ_all = []
        for result in trad_rt_results:
            if "detJ_values" in result and "detJ_timestamps" in result:
                trad_rt_detJ_all.append((result["detJ_timestamps"], result["detJ_values"]))
        
        if trad_rt_detJ_all:
            max_len = max(len(ts) for ts, _ in trad_rt_detJ_all)
            trad_rt_detJ_avg = np.zeros(max_len)
            trad_rt_detJ_std = np.zeros(max_len)
            count = np.zeros(max_len)
            
            for ts, values in trad_rt_detJ_all:
                for i, (t, v) in enumerate(zip(ts, values)):
                    if i < max_len:
                        trad_rt_detJ_avg[i] += v
                        trad_rt_detJ_std[i] += v**2
                        count[i] += 1
            
            trad_rt_detJ_avg /= count
            trad_rt_detJ_std = np.sqrt(np.maximum(0, trad_rt_detJ_std / count - trad_rt_detJ_avg**2))
            trad_rt_timestamps = np.arange(max_len)
            
            ax.plot(trad_rt_timestamps, trad_rt_detJ_avg, label="实时传统方法", color="#3498db", linewidth=2)
            ax.fill_between(trad_rt_timestamps, trad_rt_detJ_avg - trad_rt_detJ_std, 
                          trad_rt_detJ_avg + trad_rt_detJ_std, alpha=0.3, color="#3498db")
    
    # 处理延迟传统方法
    if trad_del_results and len(trad_del_results) > 0 and "detJ_values" in trad_del_results[0]:
        trad_del_detJ_all = []
        for result in trad_del_results:
            if "detJ_values" in result and "detJ_timestamps" in result:
                trad_del_detJ_all.append((result["detJ_timestamps"], result["detJ_values"]))
        
        if trad_del_detJ_all:
            max_len = max(len(ts) for ts, _ in trad_del_detJ_all)
            trad_del_detJ_avg = np.zeros(max_len)
            trad_del_detJ_std = np.zeros(max_len)
            count = np.zeros(max_len)
            
            for ts, values in trad_del_detJ_all:
                for i, (t, v) in enumerate(zip(ts, values)):
                    if i < max_len:
                        trad_del_detJ_avg[i] += v
                        trad_del_detJ_std[i] += v**2
                        count[i] += 1
            
            trad_del_detJ_avg /= count
            trad_del_detJ_std = np.sqrt(np.maximum(0, trad_del_detJ_std / count - trad_del_detJ_avg**2))
            trad_del_timestamps = np.arange(max_len)
            
            ax.plot(trad_del_timestamps, trad_del_detJ_avg, label="延迟传统方法", color="#9b59b6", linewidth=2)
            ax.fill_between(trad_del_timestamps, trad_del_detJ_avg - trad_del_detJ_std, 
                          trad_del_detJ_avg + trad_del_detJ_std, alpha=0.3, color="#9b59b6")
    
    # 处理Stackelberg方法
    if stack_results and len(stack_results) > 0 and "detJ_values" in stack_results[0]:
        stack_detJ_all = []
        for result in stack_results:
            if "detJ_values" in result and "detJ_timestamps" in result:
                stack_detJ_all.append((result["detJ_timestamps"], result["detJ_values"]))
        
        if stack_detJ_all:
            max_len = max(len(ts) for ts, _ in stack_detJ_all)
            stack_detJ_avg = np.zeros(max_len)
            stack_detJ_std = np.zeros(max_len)
            count = np.zeros(max_len)
            
            for ts, values in stack_detJ_all:
                for i, (t, v) in enumerate(zip(ts, values)):
                    if i < max_len:
                        stack_detJ_avg[i] += v
                        stack_detJ_std[i] += v**2
                        count[i] += 1
            
            stack_detJ_avg /= count
            stack_detJ_std = np.sqrt(np.maximum(0, stack_detJ_std / count - stack_detJ_avg**2))
            stack_timestamps = np.arange(max_len)
            
            ax.plot(stack_timestamps, stack_detJ_avg, label="Stackelberg方法", 
                   color="#e74c3c", linewidth=2)
            ax.fill_between(stack_timestamps, stack_detJ_avg - stack_detJ_std, 
                          stack_detJ_avg + stack_detJ_std, alpha=0.3, color="#e74c3c")
    
    ax.set_xlabel("时间步")
    ax.set_ylabel("Fisher信息矩阵 detJ值")
    ax.set_title(f"{title_text}\nFisher信息矩阵detJ值随时间变化对比（三列）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{save_path}/detJ_evolution_delay_{delay}_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ detJ演化图: {output_path}")
    plt.close()


def plot_summary_table_for_delay_three_columns(trad_rt_stats, trad_del_stats, stack_stats, delay, save_path, timestamp, 
                                  trad_rt_results=None, trad_del_results=None, stack_results=None):
    """为单个时延场景生成对比总结表格图（三列对比）"""
    title_text = get_delay_title_text(delay)
    
    # 基础指标
    base_metrics = ["avg_detJ", "sum_rate", "Ec"]
    
    table_data = []
    
    for metric in base_metrics:
        if (metric in trad_rt_stats and metric in trad_del_stats and metric in stack_stats):
            # 获取数据（实时传统 / 延迟传统 / Stackelberg）
            trad_rt_val = trad_rt_stats[metric]
            trad_del_val = trad_del_stats[metric]
            stack_val = stack_stats[metric]
            
            # 兼容处理
            if isinstance(trad_rt_val, dict) and "mean" in trad_rt_val:
                trad_rt_mean = trad_rt_val["mean"]
            else:
                trad_rt_mean = float(trad_rt_val)
                
            if isinstance(trad_del_val, dict) and "mean" in trad_del_val:
                trad_del_mean = trad_del_val["mean"]
            else:
                trad_del_mean = float(trad_del_val)
                
            if isinstance(stack_val, dict) and "mean" in stack_val:
                stack_mean = stack_val["mean"]
            else:
                stack_mean = float(stack_val)
            
            # 获取中文名称
            if metric == "avg_detJ": metric_name = "平均Fisher信息矩阵"
            elif metric == "sum_rate": metric_name = "总数据率"
            elif metric == "Ec": metric_name = "平均能耗"
            else: metric_name = metric
            
            # 数值格式
            if metric == "avg_detJ":
                trad_rt_str = f"{trad_rt_mean:.4e}"
                trad_del_str = f"{trad_del_mean:.4e}"
                stack_str = f"{stack_mean:.4e}"
            else:
                trad_rt_str = f"{trad_rt_mean:.4f}"
                trad_del_str = f"{trad_del_mean:.4f}"
                stack_str = f"{stack_mean:.4f}"

            # 相对“延迟传统方法”的比较（百分比）
            # 正号表示数值比“延迟传统方法”更大，负号表示更小
            if trad_del_mean != 0:
                rt_vs_del = (trad_rt_mean - trad_del_mean) / abs(trad_del_mean) * 100
                stack_vs_del = (stack_mean - trad_del_mean) / abs(trad_del_mean) * 100
            else:
                rt_vs_del = 0.0
                stack_vs_del = 0.0

            rt_vs_del_str = f"{rt_vs_del:+.2f}%"
            stack_vs_del_str = f"{stack_vs_del:+.2f}%"
            
            table_data.append([
                metric_name, trad_rt_str, trad_del_str, stack_str, rt_vs_del_str, stack_vs_del_str
            ])
    
    # 从results中计算每个AUV的追踪误差
    if trad_rt_results and trad_del_results and stack_results:
        # 确定AUV数量
        n_auv = 0
        for r in trad_rt_results:
            if "avg_tracking_error" in r and isinstance(r["avg_tracking_error"], list):
                n_auv = len(r["avg_tracking_error"])
                break
        
        if n_auv > 0:
            # 为每个AUV计算平均追踪误差
            for auv_idx in range(n_auv):
                trad_rt_errors = []
                trad_del_errors = []
                stack_errors = []
                
                for r in trad_rt_results:
                    if "avg_tracking_error" in r and isinstance(r["avg_tracking_error"], list):
                        if auv_idx < len(r["avg_tracking_error"]):
                            trad_rt_errors.append(r["avg_tracking_error"][auv_idx])
                
                for r in trad_del_results:
                    if "avg_tracking_error" in r and isinstance(r["avg_tracking_error"], list):
                        if auv_idx < len(r["avg_tracking_error"]):
                            trad_del_errors.append(r["avg_tracking_error"][auv_idx])
                
                for r in stack_results:
                    if "avg_tracking_error" in r and isinstance(r["avg_tracking_error"], list):
                        if auv_idx < len(r["avg_tracking_error"]):
                            stack_errors.append(r["avg_tracking_error"][auv_idx])
                
                if trad_rt_errors and trad_del_errors and stack_errors:
                    trad_rt_mean = np.mean(trad_rt_errors)
                    trad_del_mean = np.mean(trad_del_errors)
                    stack_mean = np.mean(stack_errors)

                    if trad_del_mean != 0:
                        rt_vs_del = (trad_rt_mean - trad_del_mean) / abs(trad_del_mean) * 100
                        stack_vs_del = (stack_mean - trad_del_mean) / abs(trad_del_mean) * 100
                    else:
                        rt_vs_del = 0.0
                        stack_vs_del = 0.0

                    rt_vs_del_str = f"{rt_vs_del:+.2f}%"
                    stack_vs_del_str = f"{stack_vs_del:+.2f}%"
                    
                    table_data.append([
                        f"AUV{auv_idx} 追踪误差",
                        f"{trad_rt_mean:.4f}",
                        f"{trad_del_mean:.4f}",
                        f"{stack_mean:.4f}",
                        rt_vs_del_str,
                        stack_vs_del_str,
                    ])
    
    if not table_data:
        return
    
    fig, ax = plt.subplots(figsize=(16, len(table_data) * 0.8 + 1))
    ax.axis("tight")
    ax.axis("off")
    
    # 每行包含：指标, 实时传统, 延迟传统, Stackelberg, 实时-延迟(%), Stack-延迟(%)
    display_data = [row[:6] for row in table_data]
    
    table = ax.table(cellText=display_data,
                    colLabels=[
                        "指标",
                        "实时传统方法",
                        "延迟传统方法",
                        "Stackelberg方法",
                        "实时-延迟(%)",
                        "Stack-延迟(%)",
                    ],
                    cellLoc="center", loc="center",
                    colWidths=[0.22, 0.16, 0.16, 0.16, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(6):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # 为“Stack-延迟(%)”这一列按预期加红绿底色
    # 预期：
    # - 平均Fisher信息矩阵: Stack-延迟 < 0 （下降）为绿色，否则红色
    # - 总数据率: Stack-延迟 > 0 （上升）为绿色，否则红色
    # - 平均能耗: Stack-延迟 > 0 （上升）为绿色，否则红色
    # - 追踪误差: Stack-延迟 < 0 （下降）为绿色，否则红色
    for i, row in enumerate(table_data, start=1):
        metric_name = row[0]
        stack_vs_del_str = row[5]  # 形如 "+12.34%"
        try:
            val = float(stack_vs_del_str.replace("%", ""))
        except Exception:
            continue

        # 判断该行的预期方向
        if "Fisher" in metric_name or "Fisher信息矩阵" in metric_name:
            # detJ 期望减小
            is_good = val < 0
        elif "总数据率" in metric_name:
            is_good = val > 0
        elif "平均能耗" in metric_name:
            is_good = val > 0
        elif "追踪误差" in metric_name:
            is_good = val < 0
        else:
            # 其它指标默认：越大越好
            is_good = val > 0

        cell = table[(i, 5)]
        cell.set_facecolor("#d5f4e6" if is_good else "#ffe5e5")
    
    plt.title(f"{title_text}\n三种方法对比总结", 
             fontsize=14, fontweight="bold", pad=20)
    
    output_path = f"{save_path}/comparison_summary_delay_{delay}_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ 对比总结表: {output_path}")
    plt.close()


def plot_detailed_comparison_for_delay(delay_data, delay, save_path, timestamp):
    """为单个时延场景生成详细对比图表（支持三列对比）"""
    # 兼容新旧格式
    if "traditional_realtime" in delay_data and "traditional_delayed" in delay_data:
        # 新格式：三列对比
        trad_rt_results = delay_data["traditional_realtime"].get("results", [])
        trad_del_results = delay_data["traditional_delayed"].get("results", [])
        stack_results = delay_data["stackelberg"].get("results", [])
        trad_rt_stats = delay_data["traditional_realtime"]["stats"]
        trad_del_stats = delay_data["traditional_delayed"]["stats"]
        stack_stats = delay_data["stackelberg"]["stats"]
        
        # 1. detJ演化图（需要results数据）
        if trad_rt_results or trad_del_results or stack_results:
            plot_detJ_evolution_for_delay_three_columns(trad_rt_results, trad_del_results, stack_results, delay, save_path, timestamp)
        
        # 2. 对比总结表（传入results以计算每个AUV的追踪误差）
        plot_summary_table_for_delay_three_columns(trad_rt_stats, trad_del_stats, stack_stats, delay, save_path, timestamp,
                                      trad_rt_results=trad_rt_results, trad_del_results=trad_del_results, 
                                      stack_results=stack_results)
    else:
        # 旧格式：两列对比（向后兼容）
        trad_results = delay_data.get("traditional", {}).get("results", [])
        stack_results = delay_data.get("stackelberg", {}).get("results", [])
        trad_stats = delay_data.get("traditional", {}).get("stats", {})
        stack_stats = delay_data.get("stackelberg", {}).get("stats", {})
        
        # 1. detJ演化图（需要results数据）
        if trad_results or stack_results:
            plot_detJ_evolution_for_delay(trad_results, stack_results, delay, save_path, timestamp)
        
        # 2. 对比总结表（传入results以计算每个AUV的追踪误差）
        plot_summary_table_for_delay(trad_stats, stack_stats, delay, save_path, timestamp,
                                      trad_results=trad_results, stack_results=stack_results)


def find_most_representative_result(results, metric="avg_detJ"):
    """
    找到最具代表性的实验结果（最接近平均值的那一次）
    """
    if not results or len(results) == 0:
        return None, 0
    
    if len(results) == 1:
        return results[0], 0
    
    values = []
    for res in results:
        if metric in res:
            val = res[metric]
            if isinstance(val, list):
                val = np.mean(val)
            values.append(float(val))
        else:
            values.append(0)
    
    if not values or all(v == 0 for v in values):
        return results[0], 0
    
    mean_value = np.mean(values)
    distances = [abs(v - mean_value) for v in values]
    best_idx = np.argmin(distances)
    
    print(f"  📊 最具代表性实验: 第 {best_idx + 1} 次 ({metric}={values[best_idx]:.4f}, 平均={mean_value:.4f})")
    
    return results[best_idx], best_idx


def plot_trajectory(results, title_suffix, output_dir, timestamp=None):
    """绘制轨迹图 (基于draw_trajectory.py)"""
    if not results:
        return
        
    # 找到最具代表性的实验结果
    res, idx = find_most_representative_result(results, metric="avg_detJ")
    if res is None:
        return
    
    # 尝试从title_suffix解析delay值，以生成正确的标题
    display_title = title_suffix
    try:
        # title_suffix 格式如: "Trad_Delay_0.0" 或 "Stack_Delay_1.0"
        parts = title_suffix.split('_')
        if len(parts) >= 3 and parts[-2] == "Delay":
            delay_val = float(parts[-1])
            method = parts[0]
            
            # 使用新的标题生成函数
            delay_text = get_delay_title_text(delay_val)
            method_text = "Stackelberg" if "Stack" in method else "传统方法"
            
            display_title = f"{method_text} - {delay_text}"
    except:
        pass
    
    x_auv = res.get("x_auv")
    y_auv = res.get("y_auv")
    x_usv = res.get("x_usv")
    y_usv = res.get("y_usv")
    SoPcenter = np.array(res.get("SoPcenter")) if res.get("SoPcenter") else None
    lda = res.get("lda")
    
    if not x_auv or SoPcenter is None:
        return

    N_AUV = len(x_auv)
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(linestyle="--", color="#cccccc")
    ax.set_aspect(1)
    
    P_length = len(lda) // 2
    X_length = len(lda) - P_length
    p_shape = ["P"] * P_length + ["X"] * X_length
    p_size = [48] * P_length + [45] * X_length
    p_color = ["indianred", "darkorange", "lightseagreen", "darkviolet", "gainsboro"]
    if len(lda) > 5:
        p_color += list(cm.rainbow(np.linspace(0.05, 0.95, len(lda) - 5)))
    
    U2_shape = ["d", "s"]
    U2_color = ["mediumorchid", "salmon"]
    if len(x_auv) > 2:
        U2_color += list(cm.nipy_spectral(np.linspace(0.1, 0.9, len(x_auv) - 2)))
    U2_color.append("limegreen")
    
    lw_SNpoint = 0.7
    lw_U2point = 1.1
    lw_line = 1.8
    
    lmda = np.sort(np.unique(np.array(lda)))
    fig_obj_SNs = []
    
    for i, l in enumerate(lmda):
        SN_xy = SoPcenter[np.array(lda) == l]
        sn_obj = ax.scatter(
            SN_xy[:, 0],
            SN_xy[:, 1],
            marker=p_shape[i] if i < len(p_shape) else "o",
            s=p_size[i] if i < len(p_size) else 40,
            color=p_color[i] if i < len(p_color) else "gray",
            label=f"SN λ={l}",
            edgecolors="k",
            linewidths=lw_SNpoint,
        )
        fig_obj_SNs.append(sn_obj)
        
    fig_obj_lines = []
    fig_obj_startpoint = []
    fig_obj_endpoint = []
    
    for i in range(N_AUV):
        draw_x = x_auv[i][::4]
        draw_y = y_auv[i][::4]
        
        (line_obj,) = ax.plot(
            draw_x,
            draw_y,
            linestyle="--",
            linewidth=lw_line,
            color=U2_color[i] if i < len(U2_color) else "blue",
        )
        sp_obj = ax.scatter(
            draw_x[0], draw_y[0],
            marker=U2_shape[0], color=U2_color[i] if i < len(U2_color) else "blue",
            edgecolors="k", linewidths=lw_U2point,
        )
        ep_obj = ax.scatter(
            draw_x[-1], draw_y[-1],
            marker=U2_shape[1], color=U2_color[i] if i < len(U2_color) else "blue",
            edgecolors="k", linewidths=lw_U2point,
        )
        fig_obj_lines.append(line_obj)
        fig_obj_startpoint.append(sp_obj)
        fig_obj_endpoint.append(ep_obj)
        
    draw_x_usv = x_usv[::4]
    draw_y_usv = y_usv[::4]
    
    fig_obj_lines.append(
        ax.plot(
            draw_x_usv, draw_y_usv, 
            linestyle="--", linewidth=lw_line, color=U2_color[-1]
        )[0]
    )
    fig_obj_startpoint.append(
        ax.scatter(
            draw_x_usv[0], draw_y_usv[0],
            marker=U2_shape[0], color=U2_color[-1],
            edgecolors="k", linewidths=lw_U2point,
        )
    )
    fig_obj_endpoint.append(
        ax.scatter(
            draw_x_usv[-1], draw_y_usv[-1],
            marker=U2_shape[1], color=U2_color[-1],
            edgecolors="k", linewidths=lw_U2point,
        )
    )
    
    plt.rcParams.update({"font.size": 8})
    ax.legend(
        [tuple(fig_obj_SNs)] + fig_obj_lines + [tuple(fig_obj_startpoint), tuple(fig_obj_endpoint)],
        ["SNs priority"] + [f"AUV{i+1}" for i in range(N_AUV)] + ["USV", "Start", "End"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(1.1, 1.15),
    )
    plt.rcParams.update({"font.size": 11})
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"轨迹对比 - {display_title}")
    
    plt.tight_layout()
    if timestamp:
        filename = f"trajectory_{title_suffix}_{timestamp}.png"
    else:
        filename = f"trajectory_{title_suffix}.png"
        
    filename = filename.replace(" ", "_")
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ 轨迹图: {output_path}")
    plt.close()


def plot_tracking_error_evolution(results, exp_info, title_suffix, output_dir, timestamp=None):
    """绘制追踪误差演化图 (基于draw_tracking_error.py)"""
    if not results:
        return
    
    # 找到最具代表性的实验结果
    res, idx = find_most_representative_result(results, metric="avg_detJ")
    if res is None:
        return
    
    # 尝试从title_suffix解析delay值
    display_title = title_suffix
    try:
        # title_suffix 格式如: "Trad_Delay_0.0" 或 "Stack_Delay_1.0"
        parts = title_suffix.split('_')
        if len(parts) >= 3 and parts[-2] == "Delay":
            delay_val = float(parts[-1])
            method = parts[0]
            
            # 使用新的标题生成函数
            delay_text = get_delay_title_text(delay_val)
            method_text = "Stackelberg" if "Stack" in method else "传统方法"
            
            display_title = f"{method_text} - {delay_text}"
    except:
        pass
    
    tracking_error = res.get("tracking_error")
    x_auv = res.get("x_auv")
    y_auv = res.get("y_auv")
    
    if not tracking_error or not x_auv:
        return
        
    N_AUV = len(x_auv)
    x_max = exp_info.get("border_x", 200.0)
    y_max = exp_info.get("border_y", 200.0)
    H = exp_info.get("H", 100)
    
    usbl = USBL()
    tidewave = TideWave(H=H, X_max=x_max, Y_max=y_max, T_max=len(tracking_error[0]))
    tidewave.calc_tideWave()
    
    terror_point_o = [[] for _ in range(N_AUV)]
    terror_point_m = [[] for _ in range(N_AUV)]
    
    # 使用所有数据的最小长度，避免索引越界
    calc_len = len(tracking_error[0])
    for j in range(N_AUV):
        if len(x_auv[j]) < calc_len:
            calc_len = len(x_auv[j])
        if len(y_auv[j]) < calc_len:
            calc_len = len(y_auv[j])
    
    for i in range(calc_len):
        tide_h_o = tidewave.get_tideHeight(0, 0, i)
        tide_h_m = tidewave.get_tideHeight(0.5, 0.5, i)
        for j in range(N_AUV):
            real_auv_posit_o = np.array([x_auv[j][i], y_auv[j][i], tide_h_o])
            real_auv_posit_m = np.array(
                [x_auv[j][i] - 0.5 * x_max, y_auv[j][i] - 0.5 * y_max, tide_h_m]
            )
            pred_auv_posit_o = usbl.calcPosit(real_auv_posit_o)[:2]
            pred_auv_posit_m = usbl.calcPosit(real_auv_posit_m)[:2]
            
            terror_point_o[j].append(
                np.linalg.norm(real_auv_posit_o[:2] - pred_auv_posit_o)
            )
            terror_point_m[j].append(
                np.linalg.norm(real_auv_posit_m[:2] - pred_auv_posit_m)
            )
            
    sum_tracking_error = np.sum(tracking_error, axis=0)
    sum_terror_point_o = np.sum(np.array(terror_point_o), axis=0)
    sum_terror_point_m = np.sum(np.array(terror_point_m), axis=0)
    
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(linestyle="--", color="#cccccc")
    
    colors = ["dodgerblue", "orange", "forestgreen"]
    lw_front = 1.6
    lw_back = 0.9
    
    plt.semilogy(sum_tracking_error, color=colors[0], alpha=0.2, linewidth=lw_back)
    plt.semilogy(sum_terror_point_o, color=colors[1], alpha=0.2, linewidth=lw_back)
    plt.semilogy(sum_terror_point_m, color=colors[2], alpha=0.2, linewidth=lw_back)
    
    smooth_W = 10
    if len(sum_tracking_error) > smooth_W:
        s_error = gaussian_filter1d(sum_tracking_error, smooth_W)
        s_point_o = gaussian_filter1d(sum_terror_point_o, smooth_W)
        s_point_m = gaussian_filter1d(sum_terror_point_m, smooth_W)
    else:
        s_error = sum_tracking_error
        s_point_o = sum_terror_point_o
        s_point_m = sum_terror_point_m
        
    plt.semilogy(s_error, color=colors[0], linewidth=lw_front, label="本方法")
    plt.semilogy(s_point_o, color=colors[1], linewidth=lw_front, label="固定位置 (0,0)")
    plt.semilogy(s_point_m, color=colors[2], linewidth=lw_front, label="固定位置 (中心)")
    
    plt.xlabel("时间 (s)")
    plt.ylabel("总追踪误差 (m)")
    plt.title(f"追踪误差演化 - {display_title}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    if timestamp:
        filename = f"tracking_error_evolution_{title_suffix}_{timestamp}.png"
    else:
        filename = f"tracking_error_evolution_{title_suffix}.png"
        
    filename = filename.replace(" ", "_")
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ 追踪误差演化图: {output_path}")
    plt.close()


def generate_delay_comparison_plots(results, save_path, timestamp):
    """
    生成时延对比可视化图表（类似comparison_results中的多样化图表）
    """
    delay_scenarios = results["experiment_info"]["delay_scenarios"]
    
    print("\n生成可视化图表...")
    
    # 为每个时延场景生成详细对比图表
    exp_info = results.get("experiment_info", {})
    
    for delay in delay_scenarios:
        delay_key = f"delay_{delay}"
        if delay_key in results["results"]:
            delay_data = results["results"][delay_key]
            plot_detailed_comparison_for_delay(delay_data, delay, save_path, timestamp)
            
            # 绘制高级轨迹图和误差图
            # 兼容新旧格式
            if "traditional_realtime" in delay_data:
                # 新格式：三列对比
                trad_rt_res = delay_data["traditional_realtime"].get("results")
                trad_del_res = delay_data["traditional_delayed"].get("results")
                stack_res = delay_data["stackelberg"].get("results")
                
                if trad_rt_res:
                    plot_trajectory(trad_rt_res, f"TradRealtime_Delay_{delay}", save_path, timestamp)
                    plot_tracking_error_evolution(trad_rt_res, exp_info, f"TradRealtime_Delay_{delay}", save_path, timestamp)
                
                if trad_del_res:
                    plot_trajectory(trad_del_res, f"TradDelayed_Delay_{delay}", save_path, timestamp)
                    plot_tracking_error_evolution(trad_del_res, exp_info, f"TradDelayed_Delay_{delay}", save_path, timestamp)
                
                if stack_res:
                    plot_trajectory(stack_res, f"Stack_Delay_{delay}", save_path, timestamp)
                    plot_tracking_error_evolution(stack_res, exp_info, f"Stack_Delay_{delay}", save_path, timestamp)
            else:
                # 旧格式：两列对比（向后兼容）
                trad_res = delay_data.get("traditional", {}).get("results")
                stack_res = delay_data.get("stackelberg", {}).get("results")
                
                if trad_res:
                    plot_trajectory(trad_res, f"Trad_Delay_{delay}", save_path, timestamp)
                    plot_tracking_error_evolution(trad_res, exp_info, f"Trad_Delay_{delay}", save_path, timestamp)
                
                if stack_res:
                    plot_trajectory(stack_res, f"Stack_Delay_{delay}", save_path, timestamp)
                    plot_tracking_error_evolution(stack_res, exp_info, f"Stack_Delay_{delay}", save_path, timestamp)
            
    print("✅ 所有可视化图表已生成完成！")


def main():
    """主函数"""
    # 确定结果文件
    if args.result_file:
        result_file = args.result_file
    else:
        result_file = load_latest_result()
        if not result_file:
            print("未找到延迟对比结果文件，请先运行 compare_delay_stackelberg.py")
            return
    
    print(f"加载延迟对比结果: {result_file}")
    data = load_delay_comparison_data(result_file)
    
    # 从文件名或数据中提取时间戳
    if "experiment_info" in data and "timestamp" in data["experiment_info"]:
        timestamp = data["experiment_info"]["timestamp"]
    else:
        # 从文件名提取时间戳
        import re
        match = re.search(r'(\d{8}_\d{6})', result_file)
        if match:
            timestamp = match.group(1)
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确定输出目录：如果结果文件在子文件夹中，使用该子文件夹；否则创建新的子文件夹
    if os.path.dirname(result_file) != args.output_dir:
        # 结果文件在子文件夹中
        output_dir = os.path.dirname(result_file)
    else:
        # 结果文件在根目录，创建新的子文件夹
        output_dir = f"{args.output_dir}/delay_comparison_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    
    # 生成各种可视化图表
    generate_delay_comparison_plots(data, output_dir, timestamp)
    
    print("\n所有可视化图表已生成完成！")


if __name__ == "__main__":
    main()
