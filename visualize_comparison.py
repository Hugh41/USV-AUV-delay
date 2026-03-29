"""
可视化Stackelberg博弈 vs 传统方法的对比结果
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
    "ep_reward": "回合奖励",
    "avg_detJ": "平均Fisher信息矩阵detJ",
    "max_detJ": "最大Fisher信息矩阵detJ",
    "sum_rate": "总数据率",
    "idu": "数据更新次数",
    "Ec": "平均能耗",
    "N_DO": "数据溢出率",
    "avg_tracking_error": "平均跟踪误差",
    "avg_usv_move": "USV平均移动距离",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--result_file",
    type=str,
    default=None,
    help="对比结果文件路径（JSON或Pickle），如果不指定则使用最新的结果",
)
parser.add_argument(
    "--output_dir", type=str, default="comparison_results", help="输出目录"
)

# 只在直接运行时解析参数，作为模块导入时不解析
if __name__ == "__main__":
    args = parser.parse_args()
else:
    # 作为模块导入时，创建一个默认的args对象
    class DefaultArgs:
        result_file = None
        output_dir = "comparison_results"
    args = DefaultArgs()


def load_latest_result():
    """加载最新的对比结果"""
    result_dir = args.output_dir
    # 先查找子文件夹中的文件
    subdirs = glob(f"{result_dir}/comparison_*/")
    if subdirs:
        # 找到最新的子文件夹
        latest_dir = max(subdirs, key=os.path.getctime)
        json_files = glob(f"{latest_dir}comparison_*.json")
        if json_files:
            return max(json_files, key=os.path.getctime)
    # 如果没有子文件夹，查找根目录下的文件（向后兼容）
    json_files = glob(f"{result_dir}/comparison_*.json")
    if json_files:
        latest_file = max(json_files, key=os.path.getctime)
        return latest_file
    return None


def load_comparison_data(file_path):
    """加载对比数据"""
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


def plot_metric_comparison(data, output_dir, timestamp=None):
    """绘制指标对比图"""
    trad_stats = data["traditional"]["stats"]
    stack_stats = data["stackelberg"]["stats"]
    
    # 选择要对比的关键指标
    metrics_to_plot = [
        "ep_reward",
        "avg_detJ",
        "max_detJ",
        "sum_rate",
        "idu",
        "Ec",
        "N_DO",
    ]
    
    # 过滤存在的指标
    metrics_to_plot = [
        m for m in metrics_to_plot if m in trad_stats and m in stack_stats
    ]
    
    if not metrics_to_plot:
        print("没有可绘制的指标")
        return
    
    # 创建对比图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        trad_mean = trad_stats[metric]["mean"]
        trad_std = trad_stats[metric]["std"]
        stack_mean = stack_stats[metric]["mean"]
        stack_std = stack_stats[metric]["std"]
        
        x = np.arange(2)
        means = [trad_mean, stack_mean]
        stds = [trad_std, stack_std]
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=["#3498db", "#e74c3c"])
        ax.set_xticks(x)
        ax.set_xticklabels(["传统方法", "Stackelberg"])
        ax.set_ylabel("值")
        # 使用中文标题
        title = METRIC_NAMES_CN.get(metric, metric.replace("_", " "))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # 计算Y轴范围，用于确定标签位置
        y_max = max(means) + max(stds)
        y_range = max(means) - min(means) if max(means) != min(means) else max(means) * 0.1
        
        # 添加数值标签（黑色，放在柱状图顶部，误差棒上方）
        for i, (mean, std) in enumerate(zip(means, stds)):
            # 黑色标签放在误差棒上方一点
            label_y = mean + std + y_range * 0.05
            ax.text(i, label_y, f"{mean:.4f}", ha="center", va="bottom", 
                   fontsize=8, color="black")
        
        # 计算改进百分比
        if trad_mean != 0:
            improvement = ((stack_mean - trad_mean) / abs(trad_mean)) * 100
            color = "green" if improvement > 0 else "red"
            # 红绿百分比标签放在更高的位置，确保不与黑色标签重叠
            # 放在Stackelberg柱状图上方，至少比黑色标签高20%的Y轴范围
            percentage_y = stack_mean + stack_std + y_range * 0.25
            ax.text(
                1,
                percentage_y,
                f"{improvement:+.1f}%",
                ha="center",
                va="bottom",
                color=color,
                fontweight="bold",
                fontsize=9,
            )
    
    # 隐藏多余的子图
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    if timestamp:
        output_path = f"{output_dir}/metric_comparison_{timestamp}.png"
    else:
        output_path = f"{output_dir}/metric_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"指标对比图已保存到: {output_path}")
    plt.close()


def plot_detJ_evolution(data, output_dir, timestamp=None):
    """绘制detJ值随时间的变化"""
    trad_results = data["traditional"]["results"]
    stack_results = data["stackelberg"]["results"]
    
    # 计算平均detJ值随时间的变化
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 处理传统方法
    if trad_results and "detJ_values" in trad_results[0]:
        trad_detJ_all = []
        for result in trad_results:
            if "detJ_values" in result and "detJ_timestamps" in result:
                trad_detJ_all.append(
                    (result["detJ_timestamps"], result["detJ_values"])
                )
        
        if trad_detJ_all:
            # 找到最大时间长度
            max_len = max(len(ts) for ts, _ in trad_detJ_all)
            # 对齐时间序列
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
            trad_detJ_std = np.sqrt(trad_detJ_std / count - trad_detJ_avg**2)
            trad_timestamps = np.arange(max_len)
            
            ax.plot(
                trad_timestamps,
                trad_detJ_avg,
                label="传统方法",
                color="#3498db",
                linewidth=2,
            )
            ax.fill_between(
                trad_timestamps,
                trad_detJ_avg - trad_detJ_std,
                trad_detJ_avg + trad_detJ_std,
                alpha=0.3,
                color="#3498db",
            )
    
    # 处理Stackelberg方法
    if stack_results and "detJ_values" in stack_results[0]:
        stack_detJ_all = []
        for result in stack_results:
            if "detJ_values" in result and "detJ_timestamps" in result:
                stack_detJ_all.append(
                    (result["detJ_timestamps"], result["detJ_values"])
                )
        
        if stack_detJ_all:
            # 找到最大时间长度
            max_len = max(len(ts) for ts, _ in stack_detJ_all)
            # 对齐时间序列
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
            stack_detJ_std = np.sqrt(stack_detJ_std / count - stack_detJ_avg**2)
            stack_timestamps = np.arange(max_len)
            
            ax.plot(
                stack_timestamps,
                stack_detJ_avg,
                label="Stackelberg方法",
                color="#e74c3c",
                linewidth=2,
            )
            ax.fill_between(
                stack_timestamps,
                stack_detJ_avg - stack_detJ_std,
                stack_detJ_avg + stack_detJ_std,
                alpha=0.3,
                color="#e74c3c",
            )
    
    ax.set_xlabel("时间步")
    ax.set_ylabel("Fisher信息矩阵 detJ值")
    ax.set_title("Fisher信息矩阵detJ值随时间变化对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if timestamp:
        output_path = f"{output_dir}/detJ_evolution_{timestamp}.png"
    else:
        output_path = f"{output_dir}/detJ_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"detJ演化图已保存到: {output_path}")
    plt.close()


def plot_tracking_error_comparison(data, output_dir, timestamp=None):
    """绘制跟踪误差对比"""
    trad_results = data["traditional"]["results"]
    stack_results = data["stackelberg"]["results"]
    
    if not trad_results or not stack_results:
        return
    
    # 提取跟踪误差数据
    trad_errors = []
    stack_errors = []
    
    for result in trad_results:
        if "avg_tracking_error" in result:
            trad_errors.append(result["avg_tracking_error"])
    
    for result in stack_results:
        if "avg_tracking_error" in result:
            stack_errors.append(result["avg_tracking_error"])
    
    if not trad_errors or not stack_errors:
        return
    
    # 计算每个AUV的平均跟踪误差
    n_auv = len(trad_errors[0]) if trad_errors else len(stack_errors[0])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(n_auv)
    width = 0.35
    
    trad_means = [np.mean([err[i] for err in trad_errors]) for i in range(n_auv)]
    trad_stds = [np.std([err[i] for err in trad_errors]) for i in range(n_auv)]
    stack_means = [np.mean([err[i] for err in stack_errors]) for i in range(n_auv)]
    stack_stds = [np.std([err[i] for err in stack_errors]) for i in range(n_auv)]
    
    bars1 = ax.bar(
        x - width / 2,
        trad_means,
        width,
        yerr=trad_stds,
        label="传统方法",
        alpha=0.7,
        color="#3498db",
        capsize=5,
    )
    bars2 = ax.bar(
        x + width / 2,
        stack_means,
        width,
        yerr=stack_stds,
        label="Stackelberg方法",
        alpha=0.7,
        color="#e74c3c",
        capsize=5,
    )
    
    ax.set_xlabel("AUV索引")
    ax.set_ylabel("平均跟踪误差")
    ax.set_title("AUV跟踪误差对比")
    ax.set_xticks(x)
    ax.set_xticklabels([f"AUV {i}" for i in range(n_auv)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    if timestamp:
        output_path = f"{output_dir}/tracking_error_comparison_{timestamp}.png"
    else:
        output_path = f"{output_dir}/tracking_error_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"跟踪误差对比图已保存到: {output_path}")
    plt.close()


def plot_summary_table(data, output_dir, timestamp=None):
    """生成对比总结表格图"""
    trad_stats = data["traditional"]["stats"]
    stack_stats = data["stackelberg"]["stats"]
    trad_results = data["traditional"]["results"]
    stack_results = data["stackelberg"]["results"]
    
    # 基础指标
    base_metrics = ["avg_detJ", "sum_rate", "Ec"]
    
    # 提取AUV追踪误差
    # 需要从results中重新计算，因为stats中只有平均值
    def get_auv_errors(results, n_auvs=2):
        auv_errors = [[] for _ in range(n_auvs)]
        for res in results:
            if "avg_tracking_error" in res:
                # avg_tracking_error 是一个列表，包含每个AUV的误差
                for i in range(min(len(res["avg_tracking_error"]), n_auvs)):
                    auv_errors[i].append(res["avg_tracking_error"][i])
        
        means = []
        for errs in auv_errors:
            if errs:
                means.append(np.mean(errs))
            else:
                means.append(0.0)
        return means

    # 假设有2个AUV
    trad_auv_errors = get_auv_errors(trad_results)
    stack_auv_errors = get_auv_errors(stack_results)
    
    table_data = []
    
    # 1. 添加基础指标
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
            # 平均fisher矩阵预期减小 (Stackelberg < 传统) -> improvement < 0 为符合预期(绿)
            # 总数据率预期增大 (Stackelberg > 传统) -> improvement > 0 为符合预期(绿)
            # 平均耗能预期增大 (Stackelberg > 传统) -> improvement > 0 为符合预期(绿)
            
            if metric == "avg_detJ":
                is_good = improvement < 0
            elif metric == "sum_rate":
                is_good = improvement > 0
            elif metric == "Ec":
                is_good = improvement > 0
            else:
                # 其他指标默认越大越好符合预期吗？
                # 暂时保持默认逻辑：improvement > 0 为绿
                is_good = improvement > 0
            
            # 格式化数值
            if metric == "avg_detJ":
                trad_str = f"{trad_mean:.4e}"
                stack_str = f"{stack_mean:.4e}"
            else:
                trad_str = f"{trad_mean:.4f}"
                stack_str = f"{stack_mean:.4f}"
            
            table_data.append([
                metric_name, trad_str, stack_str, f"{improvement:+.2f}%", is_good
            ])
            
    # 2. 添加AUV追踪误差
    for i in range(len(trad_auv_errors)):
        trad_err = trad_auv_errors[i]
        stack_err = stack_auv_errors[i]
        
        if trad_err != 0:
            improvement = (stack_err - trad_err) / abs(trad_err) * 100
        else:
            improvement = 0
            
        # 追踪误差越小越好
        is_good = improvement < 0
        
        table_data.append([
            f"AUV{i} 追踪误差",
            f"{trad_err:.4f}",
            f"{stack_err:.4f}",
            f"{improvement:+.2f}%",
            is_good
        ])
    
    # 绘制表格
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.8 + 1))
    ax.axis("tight")
    ax.axis("off")
    
    display_data = [row[:4] for row in table_data]
    
    table = ax.table(
        cellText=display_data,
        colLabels=["指标", "传统方法", "Stackelberg方法", "改进"],
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.2, 0.2, 0.15],
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头
    for i in range(4):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # 设置颜色
    for i in range(1, len(table_data) + 1):
        is_good = table_data[i - 1][4]
        if is_good:
            table[(i, 3)].set_facecolor("#d5f4e6")  # 绿色
        else:
            table[(i, 3)].set_facecolor("#ffe5e5")  # 红色
            
    plt.title("Stackelberg博弈 vs 传统方法对比总结", fontsize=14, fontweight="bold", pad=20)
    
    if timestamp:
        output_path = f"{output_dir}/comparison_summary_{timestamp}.png"
    else:
        output_path = f"{output_dir}/comparison_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"对比总结表已保存到: {output_path}")
    plt.close()


def find_most_representative_result(results, metric="avg_detJ"):
    """
    找到最具代表性的实验结果（最接近平均值的那一次）
    
    Args:
        results: 所有实验结果列表
        metric: 用于比较的指标（默认用 avg_detJ）
    
    Returns:
        最接近平均值的实验结果，以及它的索引
    """
    if not results or len(results) == 0:
        return None, 0
    
    if len(results) == 1:
        return results[0], 0
    
    # 提取所有实验的指标值
    values = []
    for res in results:
        if metric in res:
            val = res[metric]
            # 处理列表类型的指标（如 avg_tracking_error）
            if isinstance(val, list):
                val = np.mean(val)
            values.append(float(val))
        else:
            values.append(0)
    
    if not values or all(v == 0 for v in values):
        return results[0], 0
    
    # 计算平均值
    mean_value = np.mean(values)
    
    # 找到最接近平均值的实验
    distances = [abs(v - mean_value) for v in values]
    best_idx = np.argmin(distances)
    
    print(f"  📊 找到最具代表性的实验: 第 {best_idx + 1} 次 (指标 {metric}={values[best_idx]:.4f}, 平均={mean_value:.4f})")
    
    return results[best_idx], best_idx


def plot_trajectory(results, title_suffix, output_dir, timestamp=None):
    """绘制轨迹图 (基于draw_trajectory.py)"""
    if not results:
        return
        
    # 找到最具代表性的实验结果（最接近平均值的那一次）
    res, idx = find_most_representative_result(results, metric="avg_detJ")
    if res is None:
        return
    
    x_auv = res.get("x_auv")
    y_auv = res.get("y_auv")
    x_usv = res.get("x_usv")
    y_usv = res.get("y_usv")
    SoPcenter = np.array(res.get("SoPcenter")) if res.get("SoPcenter") else None
    lda = res.get("lda")
    
    if not x_auv or SoPcenter is None:
        return

    N_AUV = len(x_auv)
    
    # 绘图设置
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(linestyle="--", color="#cccccc")
    ax.set_aspect(1)
    
    # 配色方案
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
    U2_color.append("limegreen") # USV color
    
    lw_SNpoint = 0.7
    lw_U2point = 1.1
    lw_line = 1.8
    
    lmda = np.sort(np.unique(np.array(lda)))
    fig_obj_SNs = []
    
    # 绘制传感器节点
    for idx, l in enumerate(lmda):
        SN_xy = SoPcenter[np.array(lda) == l]
        sn_obj = ax.scatter(
            SN_xy[:, 0],
            SN_xy[:, 1],
            marker=p_shape[idx],
            s=p_size[idx],
            color=p_color[idx],
            label=f"SN λ={l}",
            edgecolors="k",
            linewidths=lw_SNpoint,
        )
        fig_obj_SNs.append(sn_obj)
        
    fig_obj_lines = []
    fig_obj_startpoint = []
    fig_obj_endpoint = []
    
    # 绘制AUV轨迹
    for i in range(N_AUV):
        # 简化轨迹（每隔4个点取一个，提高可读性）
        draw_x = x_auv[i][::4]
        draw_y = y_auv[i][::4]
        
        (line_obj,) = ax.plot(
            draw_x,
            draw_y,
            linestyle="--",
            linewidth=lw_line,
            color=U2_color[i],
        )
        sp_obj = ax.scatter(
            draw_x[0], draw_y[0],
            marker=U2_shape[0], color=U2_color[i],
            edgecolors="k", linewidths=lw_U2point,
        )
        ep_obj = ax.scatter(
            draw_x[-1], draw_y[-1],
            marker=U2_shape[1], color=U2_color[i],
            edgecolors="k", linewidths=lw_U2point,
        )
        fig_obj_lines.append(line_obj)
        fig_obj_startpoint.append(sp_obj)
        fig_obj_endpoint.append(ep_obj)
        
    # 绘制USV轨迹
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
    
    # 图例
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
    ax.set_title(f"轨迹对比 - {title_suffix}")
    
    plt.tight_layout()
    if timestamp:
        filename = f"trajectory_{title_suffix}_{timestamp}.png"
    else:
        filename = f"trajectory_{title_suffix}.png"
        
    # 清理文件名中的非法字符
    filename = filename.replace(" ", "_")
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"轨迹图已保存到: {output_path}")
    plt.close()


def plot_tracking_error_evolution(results, exp_info, title_suffix, output_dir, timestamp=None):
    """绘制追踪误差演化图 (基于draw_tracking_error.py)"""
    if not results:
        return
    
    # 找到最具代表性的实验结果
    res, idx = find_most_representative_result(results, metric="avg_detJ")
    if res is None:
        return
    
    tracking_error = res.get("tracking_error")
    x_auv = res.get("x_auv")
    y_auv = res.get("y_auv")
    
    if not tracking_error or not x_auv:
        return
        
    N_AUV = len(x_auv)
    x_max = exp_info.get("border_x", 200.0)
    y_max = exp_info.get("border_y", 200.0)
    H = exp_info.get("H", 100)
    
    # 计算理论误差 (Baseline)
    usbl = USBL()
    tidewave = TideWave(H=H, X_max=x_max, Y_max=y_max, T_max=len(tracking_error[0]))
    tidewave.calc_tideWave()
    
    terror_point_o = [[] for _ in range(N_AUV)]  # (0,0)
    terror_point_m = [[] for _ in range(N_AUV)]  # midpoint
    
    # 限制计算长度，避免过长
    calc_len = len(tracking_error[0])
    
    for i in range(calc_len):
        tide_h_o = tidewave.get_tideHeight(0, 0, i)
        tide_h_m = tidewave.get_tideHeight(0.5, 0.5, i)
        for j in range(N_AUV):
            real_auv_posit_o = np.array([x_auv[j][i], y_auv[j][i], tide_h_o])
            real_auv_posit_m = np.array(
                [x_auv[j][i] - 0.5 * x_max, y_auv[j][i] - 0.5 * y_max, tide_h_m]
            )
            # calc posit
            pred_auv_posit_o = usbl.calcPosit(real_auv_posit_o)[:2]
            pred_auv_posit_m = usbl.calcPosit(real_auv_posit_m)[:2]
            
            terror_point_o[j].append(
                np.linalg.norm(real_auv_posit_o[:2] - pred_auv_posit_o)
            )
            terror_point_m[j].append(
                np.linalg.norm(real_auv_posit_m[:2] - pred_auv_posit_m)
            )
            
    # Sum error across AUVs
    sum_tracking_error = np.sum(tracking_error, axis=0)
    sum_terror_point_o = np.sum(np.array(terror_point_o), axis=0)
    sum_terror_point_m = np.sum(np.array(terror_point_m), axis=0)
    
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(linestyle="--", color="#cccccc")
    
    colors = ["dodgerblue", "orange", "forestgreen"]
    lw_front = 1.6
    lw_back = 0.9
    
    # Draw background (raw)
    plt.semilogy(sum_tracking_error, color=colors[0], alpha=0.2, linewidth=lw_back)
    plt.semilogy(sum_terror_point_o, color=colors[1], alpha=0.2, linewidth=lw_back)
    plt.semilogy(sum_terror_point_m, color=colors[2], alpha=0.2, linewidth=lw_back)
    
    # Draw foreground (smoothed)
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
    plt.title(f"追踪误差演化 - {title_suffix}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    if timestamp:
        filename = f"tracking_error_evolution_{title_suffix}_{timestamp}.png"
    else:
        filename = f"tracking_error_evolution_{title_suffix}.png"
        
    filename = filename.replace(" ", "_")
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"追踪误差演化图已保存到: {output_path}")
    plt.close()


def main():
    """主函数"""
    # 确定结果文件
    if args.result_file:
        result_file = args.result_file
    else:
        result_file = load_latest_result()
        if not result_file:
            print("未找到对比结果文件，请先运行 compare_stackelberg.py")
            return
    
    print(f"加载对比结果: {result_file}")
    data = load_comparison_data(result_file)
    
    # 从数据或文件名中提取时间戳
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
    
    # 确定输出目录：如果结果文件在子文件夹中，使用该子文件夹；否则使用指定的输出目录
    if os.path.dirname(result_file) != args.output_dir:
        # 结果文件在子文件夹中
        output_dir = os.path.dirname(result_file)
    else:
        # 结果文件在根目录，创建新的子文件夹
        output_dir = f"{args.output_dir}/comparison_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    
    # 生成各种可视化图表
    print("\n生成可视化图表...")
    
    # 1. 核心统计图表
    plot_detJ_evolution(data, output_dir, timestamp)
    plot_summary_table(data, output_dir, timestamp)
    
    # 2. 高级可视化图表 (轨迹与误差演化)
    exp_info = data.get("experiment_info", {})
    
    # 绘制传统方法
    if "results" in data["traditional"] and data["traditional"]["results"]:
        plot_trajectory(data["traditional"]["results"], "Traditional", output_dir, timestamp)
        plot_tracking_error_evolution(data["traditional"]["results"], exp_info, "Traditional", output_dir, timestamp)
        
    # 绘制Stackelberg方法
    if "results" in data["stackelberg"] and data["stackelberg"]["results"]:
        plot_trajectory(data["stackelberg"]["results"], "Stackelberg", output_dir, timestamp)
        plot_tracking_error_evolution(data["stackelberg"]["results"], exp_info, "Stackelberg", output_dir, timestamp)
    
    print("\n所有可视化图表已生成完成！")


if __name__ == "__main__":
    main()

