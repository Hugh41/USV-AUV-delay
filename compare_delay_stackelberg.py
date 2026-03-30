"""
不同时延下 Stackelberg博弈 vs 传统方法对比实验

该脚本用于在不同通信延迟条件下，对比Stackelberg博弈方法和传统优化方法的效果
支持 TD3 和 DSAC 两种模型
"""

import math
import os
import sys
from env import Env
import numpy as np
import argparse
import copy
import pickle
import json
import pickle
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import torch

# pytorch
from td3 import TD3

# 添加DSAC路径
dsac_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DSAC-v2')
sys.path.insert(0, dsac_path)


class DSACInferenceAgent:
    """DSAC模型推理Agent（仅用于评估，不需要训练功能）"""
    
    def __init__(self, **kwargs):
        self.state_dim = kwargs.get("obsv_dim")
        self.action_dim = kwargs.get("action_dim")
        self.kwargs = kwargs
        self.networks = None
        self.use_gpu = kwargs.get("use_gpu", False)
        
    def _create_networks(self):
        """创建网络结构"""
        if self.networks is not None:
            return
            
        # 导入DSAC模块
        alg_name = self.kwargs.get("algorithm", "DSAC_V2")
        alg_file_name = alg_name.lower()
        module = __import__(alg_file_name)
        ApproxContainer = getattr(module, "ApproxContainer")
        
        # 创建网络
        self.networks = ApproxContainer(**self.kwargs)
        
        if self.use_gpu and torch.cuda.is_available():
            self.networks.cuda()
    
    def select_action(self, obs, deterministic=True):
        """选择动作（默认使用确定性策略）"""
        self._create_networks()
        
        with torch.no_grad():
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            if self.use_gpu:
                batch_obs = batch_obs.cuda()
            
            logits = self.networks.policy(batch_obs)
            action_dist = self.networks.create_action_distributions(logits)
            
            if deterministic:
                action = action_dist.mode()
            else:
                action, _ = action_dist.sample()
            
            action = action.detach().cpu().numpy()[0]
        return action
    
    def load(self, save_path, episode, idx=0):
        """加载DSAC模型"""
        self._create_networks()
        
        model_file = f"{save_path}DSAC_{idx}_{episode}.pkl"
        if os.path.exists(model_file):
            self.networks.load_state_dict(torch.load(model_file, map_location='cpu'))
            print(f"  ✓ 加载DSAC模型: {model_file}")
        else:
            raise FileNotFoundError(f"DSAC模型文件不存在: {model_file}")

# 导入丢包模型（根据PDF文档）
try:
    from water_model import get_package_loss
    PACKET_LOSS_AVAILABLE = True
except ImportError:
    PACKET_LOSS_AVAILABLE = False
    print("⚠️  water_model模块未找到，丢包功能将不可用")

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

# 参数设置
parser = argparse.ArgumentParser()
# ------ 实验参数 ------
parser.add_argument("--repeat_num", type=int, default=50, help="每个时延条件的重复实验次数")
parser.add_argument(
    "--episode_length", type=int, default=1000, help="每个episode的长度 (sec)"
)
parser.add_argument("--load_ep", type=int, default=575, help="加载的模型episode")
parser.add_argument("--skip_model_check", action="store_true", help="跳过模型检查")
parser.add_argument("--load_existing", type=str, default=None, help="加载已存在的结果文件（JSON文件路径），跳过运行直接可视化")
parser.add_argument("--model_type", type=str, default="td3", choices=["td3", "dsac"],
                    help="模型类型: td3 (models_td3) 或 dsac (models_dsac)")
# ------ 时延参数 ------
parser.add_argument("--packet_loss_modes", type=str, default="0,1", 
                help="丢包模式列表（0=无丢包，1=有丢包），用逗号分隔")
parser.add_argument("--fixed_delay", type=float, default=0.1, 
                help="固定处理延迟（秒），默认0.1s (50ms+50ms)")
parser.add_argument("--sampling_delay_max", type=float, default=0.333, 
                help="最大采样延迟（秒）")
# ------ 环境参数 ------
parser.add_argument("--R_dc", type=float, default=6.0, help="数据收集半径")
parser.add_argument("--border_x", type=float, default=200.0, help="区域x大小")
parser.add_argument("--border_y", type=float, default=200.0, help="区域y大小")
parser.add_argument("--n_s", type=int, default=30, help="传感器节点数量")
parser.add_argument("--N_AUV", type=int, default=2, help="AUV数量")
parser.add_argument("--Q", type=float, default=2, help="传感器节点容量 (Mbits)")
parser.add_argument("--alpha", type=float, default=0.05, help="传感器选择距离优先级")
# ------ 训练参数（用于加载模型） ------
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.97)
parser.add_argument("--tau", type=float, default=0.001)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--replay_capa", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--policy_freq", type=int, default=2)
parser.add_argument(
    "--usv_update_frequency", type=int, default=5, 
    help="USV update frequency: USV updates every N steps (Leader慢，Follower快). Default: 5"
)

args = parser.parse_args()
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# 根据模型类型、AUV数量和更新频率选择保存路径
# 新格式: models_td3_{N_AUV}AUV_{update_frequency}/ 或 models_dsac_{N_AUV}AUV_{update_frequency}/
N_AUV_FOR_PATH = args.N_AUV
UPDATE_FREQ = args.usv_update_frequency
SAVE_PATH_TD3 = BASE_PATH + f"/models_td3_{N_AUV_FOR_PATH}AUV_{UPDATE_FREQ}/"
SAVE_PATH_DSAC = BASE_PATH + f"/models_dsac_{N_AUV_FOR_PATH}AUV_{UPDATE_FREQ}/"
SAVE_PATH = SAVE_PATH_DSAC if args.model_type == "dsac" else SAVE_PATH_TD3
RES_PATH = BASE_PATH + "/delay_comparison_results"
if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)


def calculate_detJ(env):
    """计算当前时刻的Fisher信息矩阵detJ值"""
    try:
        neg_detJ = env.calcnegdetJ_USV(env.usv_xy)
        detJ = -neg_detJ
        detJ = max(0, detJ)
        return detJ
    except Exception as e:
        print(f"计算detJ时出错: {e}")
        return 0


def apply_delay_to_position(old_xy, current_xy, delay_value):
    """
    根据延迟值调整位置（加权融合）
    
    Args:
        old_xy: 上一时刻位置
        current_xy: 当前真实位置
        delay_value: 延迟值（秒）
    
    Returns:
        delayed_xy: 延迟后的位置
    """
    # 延迟越大，旧位置权重越高
    # 限制延迟权重在[0, 1]范围内
    delay_weight = min(delay_value, 1.0)
    delayed_xy = delay_weight * old_xy + (1 - delay_weight) * current_xy
    return delayed_xy


def run_single_episode(env, agents, use_stackelberg=True, delay_scenario=0.0, 
                       fixed_delay=0.1, sampling_delay_max=0.333, enable_packet_loss=False):
    """
    运行单个episode并记录所有指标
    
    延迟模型（根据用户提供的图片和PDF文档）：
    - 延迟计算：传输时延 + 固定时延(0.1s) + 采样时延(0.333s * rand)
    - 丢包模型：基于water_model.py
    
    Args:
        env: 环境对象
        agents: AUV智能体列表
        use_stackelberg: 是否使用Stackelberg博弈
        delay_scenario: 这里复用为丢包模式指示器 (0=关闭丢包, 1=开启丢包)
        fixed_delay: 固定处理延迟
        sampling_delay_max: 最大采样延迟
        enable_packet_loss: (废弃，由delay_scenario控制)
        
    Returns:
        metrics: 包含所有指标的字典
    """
    # 设置是否使用Stackelberg博弈
    if hasattr(env, 'use_stackelberg'):
        env.use_stackelberg = use_stackelberg
    
    # 初始化延迟相关变量
    old_xy = np.zeros((env.N_AUV, 2))
    
    state_c = env.reset()
    state = copy.deepcopy(state_c)
    old_xy = copy.deepcopy(env.xy)  # 初始化旧位置（用于延迟模拟）
    
    # 记录轨迹
    x_auv = [[env.xy[i][0]] for i in range(env.N_AUV)]
    y_auv = [[env.xy[i][1]] for i in range(env.N_AUV)]
    x_usv = [env.usv_xy[0]]
    y_usv = [env.usv_xy[1]]
    tracking_error = [
        [np.linalg.norm(env.obs_xy[i] - env.xy[i])] for i in range(env.N_AUV)
    ]
    
    # 指标初始化
    ep_r = 0
    idu = 0
    N_DO = 0
    DQ = 0
    FX = [0] * env.N_AUV
    sum_rate = 0
    Ec = [0] * env.N_AUV
    Ht = [0] * env.N_AUV
    Ft = 0
    crash = 0
    mode = [0] * env.N_AUV
    ht = [0] * env.N_AUV
    hovers = [False] * env.N_AUV
    ep_reward = 0
    
    # 延迟相关统计
    total_delay = 0.0
    delay_count = 0
    total_packet_loss = 0
    packet_count = 0
    
    # Fisher信息矩阵detJ值记录
    detJ_values = [calculate_detJ(env)]
    detJ_timestamps = [0]
    
    # USV位置变化记录
    usv_position_changes = [0]  # 初始位置变化为0
    
    # 轨迹记录
    x_auv = [[] for _ in range(env.N_AUV)]
    y_auv = [[] for _ in range(env.N_AUV)]
    x_usv = []
    y_usv = []
    
    while True:
        act = []
        for i in range(env.N_AUV):
            iact = agents[i].select_action(state[i])
            act.append(iact)
        
        env.posit_change(act, hovers)
        state_, rewards, Done, data_rate, ec, cs = env.step_move(hovers)
        
        # 应用延迟和丢包模拟
        # 始终应用延迟（根据Prompt: Stackelberg博弈本身自带延迟）
        # 丢包由 delay_scenario 参数控制 (0=无, 1=有)
        for i in range(env.N_AUV):
            # 计算USV和AUV之间的距离
            usv_auv_diff = env.usv_xy - env.xy[i]
            distance = np.linalg.norm(usv_auv_diff)
            
            # 1. 计算延迟 (根据Prompt图片公式)
            # 传输时延(distance/1500) + 固定时延(0.1) + 采样时延(0.333*rand)
            current_delay = (distance / 1500.0) + fixed_delay + sampling_delay_max * np.random.rand()
            total_delay += current_delay
            delay_count += 1
            
            # 2. 丢包模拟
            packet_received = True
            # delay_scenario > 0.5 视为开启丢包模式
            packet_loss_mode = (delay_scenario > 0.5)
            
            if packet_loss_mode and PACKET_LOSS_AVAILABLE:
                packet_loss_rate = get_package_loss(distance)
                packet_received = (np.random.rand() > packet_loss_rate)
                total_packet_loss += (not packet_received)
                packet_count += 1
            
            # 3. 更新观测位置 (仅当收到包时)
            if packet_received:
                # 根据图片: meas_posit = delay * old + (1-delay) * new
                # 注意限制权重在[0, 1]
                weight = min(current_delay, 1.0)
                delayed_xy = weight * old_xy[i] + (1 - weight) * env.xy[i]
                env.obs_xy[i][:2] = delayed_xy[:2]
            # 如果丢包，obs_xy保持不变 (Dead Reckoning 或 Hold Last Value)
        
        # 记录detJ值
        detJ = calculate_detJ(env)
        detJ_values.append(detJ)
        detJ_timestamps.append(Ft + 1)
        
        # cs是数组，需要求和
        crash += np.sum(cs) if isinstance(cs, (np.ndarray, list)) else int(cs)
        ep_reward += np.sum(rewards) / 1000
        
        # 记录USV位置变化（与compare_stackelberg.py保持一致）
        if len(x_usv) > 0:
            prev_usv_xy = np.array([x_usv[-1], y_usv[-1]])
            current_usv_xy = env.usv_xy
            usv_move_distance = np.linalg.norm(current_usv_xy - prev_usv_xy)
            usv_position_changes.append(usv_move_distance)
        else:
            usv_position_changes.append(0)
        
        for i in range(env.N_AUV):
            # 记录轨迹（与compare_stackelberg.py保持一致）
            x_auv[i].append(env.xy[i][0])
            y_auv[i].append(env.xy[i][1])
            x_usv.append(env.usv_xy[0])
            y_usv.append(env.usv_xy[1])
            # 记录跟踪误差
            tracking_error[i].append(np.linalg.norm(env.obs_xy[i] - env.xy[i]))
            
            if mode[i] == 0:
                state[i] = copy.deepcopy(state_[i])
                if Done[i] == True:
                    idu += 1
                    ht[i] = args.Q * env.updata[i] / data_rate[i]
                    mode[i] += math.ceil(ht[i])
                    hovers[i] = True
                    sum_rate += data_rate[i]
            else:
                mode[i] -= 1
                Ht[i] += 1
                if mode[i] == 0:
                    hovers[i] = False
                    Ht[i] -= math.ceil(ht[i]) - ht[i]
                    state[i] = env.CHOOSE_AIM(idx=i, lamda=args.alpha)
        
        # 更新旧位置
        old_xy = copy.deepcopy(env.xy)
        
        Ft += 1
        env.Ft = Ft
        N_DO += env.N_DO
        FX = np.array(FX) + np.array(env.FX)
        DQ += sum(env.b_S / env.Fully_buffer)
        Ec = np.array(Ec) + np.array(ec)
        
        if Ft > args.episode_length:
            N_DO /= Ft
            DQ /= Ft
            DQ /= env.N_POI
            Ec = np.sum(np.array(Ec) / (Ft - np.array(Ht))) / env.N_AUV
            avg_delay = total_delay / delay_count if delay_count > 0 else 0.0
            avg_packet_loss_rate = total_packet_loss / packet_count if packet_count > 0 else 0.0
            
            # 计算平均detJ值
            avg_detJ = np.mean(detJ_values) if len(detJ_values) > 0 else 0
            max_detJ = np.max(detJ_values) if len(detJ_values) > 0 else 0
            min_detJ = np.min(detJ_values) if len(detJ_values) > 0 else 0
            
            # 计算USV平均移动距离
            avg_usv_move = np.mean(usv_position_changes) if len(usv_position_changes) > 0 else 0
            total_usv_move = np.sum(usv_position_changes)
            
            # 计算平均跟踪误差（列表格式，每个AUV独立统计）
            avg_tracking_error = [
                np.mean(tracking_error[i]) for i in range(env.N_AUV)
            ]
            
            # 确保所有值都是Python原生类型（字段名与comparison_results保持一致）
            metrics = {
                "ep_reward": float(ep_reward),  # 回合奖励（与comparison_results一致）
                "DQ": float(DQ),  # 数据收集率（与comparison_results一致）
                "sum_rate": float(sum_rate),  # 总数据率
                "idu": int(idu),  # 数据更新次数（与comparison_results一致）
                "Ec": float(Ec),  # 平均能耗（与comparison_results一致）
                "N_DO": float(N_DO),  # 数据溢出率（与comparison_results一致）
                "crash": int(crash),  # 碰撞次数
                "FX": [float(x) for x in FX.tolist()] if isinstance(FX, np.ndarray) else FX,  # 边界穿越（与comparison_results一致）
                "avg_detJ": float(avg_detJ),  # 平均Fisher信息矩阵detJ
                "max_detJ": float(max_detJ),  # 最大Fisher信息矩阵detJ
                "min_detJ": float(min_detJ),  # 最小Fisher信息矩阵detJ
                "avg_usv_move": float(avg_usv_move),  # USV平均移动距离
                "total_usv_move": float(total_usv_move),  # USV总移动距离
                "avg_tracking_error": avg_tracking_error,  # 平均跟踪误差（列表格式）
                "detJ_values": [float(x) for x in detJ_values],  # detJ值序列
                "detJ_timestamps": [int(x) for x in detJ_timestamps],  # detJ时间戳
                "x_auv": x_auv,  # AUV轨迹x坐标
                "y_auv": y_auv,  # AUV轨迹y坐标
                "x_usv": x_usv,  # USV轨迹x坐标
                "y_usv": y_usv,  # USV轨迹y坐标
                "tracking_error": tracking_error,  # 跟踪误差序列
                # 环境静态信息（用于可视化）
                "SoPcenter": env.SoPcenter.tolist() if isinstance(env.SoPcenter, np.ndarray) else env.SoPcenter,
                "lda": env.lda,
                # 时延相关字段（compare_delay_stackelberg.py特有）
                "avg_delay": float(avg_delay),
                "avg_packet_loss_rate": float(avg_packet_loss_rate),
                "delay_scenario": float(delay_scenario),
                "use_stackelberg": bool(use_stackelberg),
            }
            return metrics


def convert_to_native_type(obj):
    """
    转换为可序列化的格式（兼容NumPy 2.0）
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, dict):
        return {k: convert_to_native_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_type(item) for item in obj]
    elif isinstance(obj, complex) or (hasattr(np, 'complex128') and isinstance(obj, (np.complex128, np.complex64))):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    else:
        # 对于未知类型，尝试转换为字符串或直接返回
        # 如果是numpy标量，尝试转换为Python原生类型
        try:
            if hasattr(obj, 'item'):  # numpy标量有item方法
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy数组
                return obj.tolist()
        except:
            pass
        # 最后尝试直接返回，如果还是无法序列化，会在json.dump时报错
        return obj


def run_delay_comparison_experiment():
    """
    运行不同时延条件下的对比实验
    """
    print("=" * 80)
    print("不同时延下 Stackelberg博弈 vs 传统方法对比实验")
    print(f"模型类型: {args.model_type.upper()}")
    print(f"AUV数量: {args.N_AUV}")
    print(f"使用Episode: {args.load_ep}")
    print("=" * 80)
    
    # 解析丢包模式（原delay_scenarios）
    # 为了保持与可视化脚本的兼容性，我们仍然使用 delay_scenarios 这个变量名和键名
    # 但其含义已变为丢包模式：0=关闭，1=开启
    delay_scenarios = [float(x.strip()) for x in args.packet_loss_modes.split(',')]
    print(f"\n丢包模式列表: {delay_scenarios} (0=关闭, 1=开启)")
    print(f"固定处理延迟: {args.fixed_delay} 秒")
    print(f"最大采样延迟: {args.sampling_delay_max} 秒")
    print(f"每个场景重复次数: {args.repeat_num}\n")
    
    # 如果指定了加载已有文件，直接加载并可视化
    if args.load_existing:
        file_path = args.load_existing
        # 如果已经是绝对路径，直接使用；如果是相对路径，检查是否已经包含RES_PATH
        if not os.path.isabs(file_path):
            # 如果路径已经以delay_comparison_results开头，不再添加RES_PATH
            if file_path.startswith('delay_comparison_results/'):
                file_path = file_path  # 保持原样，相对于当前目录
            else:
                file_path = os.path.join(RES_PATH, args.load_existing)
        
        if os.path.exists(file_path):
            print(f"📂 加载已有结果文件: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    all_results = json.load(f)
                
                # 显示实验信息
                exp_info = all_results.get("experiment_info", {})
                n_auv = exp_info.get("N_AUV", "未知")
                load_ep = exp_info.get("load_ep", "未知")
                model_type = exp_info.get("model_type", "未知")
                print(f"   实验配置: {n_auv}个AUV, Episode {load_ep}, {model_type.upper()}模型")
                
                timestamp = exp_info.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
                # 确定结果目录：如果文件在子文件夹中，使用该子文件夹；否则创建新的
                if os.path.dirname(file_path) != RES_PATH:
                    result_dir = os.path.dirname(file_path)
                else:
                    result_dir = f"{RES_PATH}/delay_comparison_{timestamp}"
                    os.makedirs(result_dir, exist_ok=True)
                generate_delay_comparison_plots(all_results, result_dir, timestamp)
                return all_results
            except json.JSONDecodeError as e:
                print(f"❌ JSON文件格式错误（可能不完整）: {e}")
                print("   文件可能在保存时被中断，建议重新运行实验")
                return None
        else:
            print(f"❌ 文件不存在: {file_path}")
            return None
    
    # 检查模型文件
    model_exists = True
    if not args.skip_model_check:
        for i in range(args.N_AUV):
            if args.model_type == "dsac":
                model_file = f"{SAVE_PATH}DSAC_{i}_{args.load_ep}.pkl"
            else:
                model_file = f"{SAVE_PATH}TD3_{i}_{args.load_ep}_actor.pth"
            if not os.path.exists(model_file):
                model_exists = False
                print(f"❌ 模型文件不存在: {model_file}")
                break
    
    if not model_exists:
        print("\n💡 请先训练模型或使用 --skip_model_check 跳过检查")
        return
    
    # 初始化环境和智能体
    env = Env(args)
    N_AUV = args.N_AUV
    state_dim = env.state_dim
    action_dim = 2
    
    # 根据模型类型创建智能体
    if args.model_type == "dsac":
        # 创建DSAC智能体
        dsac_kwargs = {
            "algorithm": "DSAC_V2",
            "obsv_dim": state_dim,
            "action_dim": action_dim,
            "action_high_limit": np.array([1.0, 1.0], dtype=np.float32),
            "action_low_limit": np.array([-1.0, -1.0], dtype=np.float32),
            "value_func_name": "ActionValueDistri",
            "value_func_type": "MLP",
            "value_hidden_sizes": [256, 256, 256],
            "value_hidden_activation": "gelu",
            "value_output_activation": "linear",
            "value_min_log_std": -8,
            "value_max_log_std": 8,
            "policy_func_name": "StochaPolicy",
            "policy_func_type": "MLP",
            "policy_act_distribution": "TanhGaussDistribution",
            "policy_hidden_sizes": [256, 256, 256],
            "policy_hidden_activation": "gelu",
            "policy_output_activation": "linear",
            "policy_min_log_std": -20,
            "policy_max_log_std": 0.5,
            "use_gpu": False,
            "cnn_shared": False,
            "action_type": "continu",
            # 算法参数（仅用于创建网络结构，推理时不需要）
            "value_learning_rate": 0.0001,
            "policy_learning_rate": 0.0001,
            "alpha_learning_rate": 0.0003,
            "gamma": 0.97,
            "tau": 0.005,
            "auto_alpha": True,
            "alpha": 0.2,
            "delay_update": 2,
            "TD_bound": 1,
            "bound": True,
        }
        agents = [DSACInferenceAgent(**dsac_kwargs) for _ in range(N_AUV)]
    else:
        # 创建TD3智能体
        agents = [TD3(state_dim, action_dim) for _ in range(N_AUV)]
    
    # 加载模型
    print(f"\n📦 加载{args.model_type.upper()}模型文件...")
    print(f"   AUV数量: {N_AUV}")
    print(f"   Episode: {args.load_ep}")
    print(f"   模型路径: {SAVE_PATH}")
    for i in range(N_AUV):
        agents[i].load(SAVE_PATH, args.load_ep, idx=i)
    
    # 设置agents到环境中，用于Stackelberg博弈
    if hasattr(env, 'set_agents'):
        env.set_agents(agents)
    else:
        print("⚠️  警告: 环境不支持set_agents，Stackelberg博弈可能无法正常工作")
    
    # 存储所有实验结果
    # 创建以时间戳命名的子文件夹（加入model_type、AUV数量和episode信息）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    usv_freq = args.usv_update_frequency
    model_type = args.model_type
    n_auv = args.N_AUV
    load_ep = args.load_ep
    # 结果目录名包含：时间戳、更新频率、模型类型、AUV数量、Episode
    result_dir = f"{RES_PATH}/delay_comparison_{timestamp}_{usv_freq}_{model_type}_{n_auv}AUV_ep{load_ep}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 使用临时文件保存中间结果，避免数据丢失
    # 文件名也包含AUV数量和Episode信息
    temp_result_file = f"{result_dir}/delay_comparison_{timestamp}_{n_auv}AUV_ep{load_ep}.tmp"
    final_result_file = f"{result_dir}/delay_comparison_{timestamp}_{n_auv}AUV_ep{load_ep}.json"
    
    all_results = {
        "experiment_info": {
            "timestamp": timestamp,
            "delay_scenarios": delay_scenarios,
            "fixed_delay": args.fixed_delay,
            "sampling_delay_max": args.sampling_delay_max,
            "repeat_num": args.repeat_num,
            "episode_length": args.episode_length,
            "load_ep": args.load_ep,
            "N_AUV": args.N_AUV,
            "n_s": args.n_s,
            "border_x": args.border_x,
            "border_y": args.border_y,
            "model_type": args.model_type,
            "usv_update_frequency": args.usv_update_frequency,
            "H": 100,
            # 添加清晰的说明字段
            "model_info": {
                "AUV_count": args.N_AUV,
                "episode_used": args.load_ep,
                "model_type": args.model_type,
                "description": f"使用{args.N_AUV}个AUV，Episode {args.load_ep}的{args.model_type.upper()}模型"
            }
        },
        "results": {}
    }
    
    # 对每个时延场景进行实验
    
    for delay_scenario in delay_scenarios:
        mode_str = "开启" if delay_scenario > 0.5 else "关闭"
        print(f"\n{'='*60}")
        print(f"丢包模式: {mode_str} (scenario值={delay_scenario})")
        print(f"{'='*60}")
        
        traditional_results = []
        stackelberg_results = []
        
        # 运行传统方法实验
        print(f"\n运行传统方法实验（丢包={mode_str}）...")
        if hasattr(env, 'use_stackelberg'):
            env.use_stackelberg = False
        for ep in range(args.repeat_num):
            metrics = run_single_episode(
                env, agents, 
                use_stackelberg=False,
                delay_scenario=delay_scenario,
                fixed_delay=args.fixed_delay,
                sampling_delay_max=args.sampling_delay_max
            )
            print(
                f"  Traditional - Ep {ep+1}/{args.repeat_num} | "
                f"Reward: {metrics['ep_reward']:.2f} | "
                f"DetJ: {metrics['avg_detJ']:.4e} | "
                f"Rate: {metrics['sum_rate']:.2f}"
            )
            traditional_results.append(metrics)
        print(f"  ✓ 完成传统方法实验")
        
        # 运行Stackelberg方法实验
        print(f"\n运行Stackelberg方法实验（丢包={mode_str}）...")
        if hasattr(env, 'use_stackelberg'):
            env.use_stackelberg = True
        else:
            print("⚠️  警告: 环境不支持use_stackelberg属性")
            
        for ep in range(args.repeat_num):
            metrics = run_single_episode(
                env, agents,
                use_stackelberg=True,
                delay_scenario=delay_scenario,
                fixed_delay=args.fixed_delay,
                sampling_delay_max=args.sampling_delay_max
            )
            print(
                f"  Stackelberg - Ep {ep+1}/{args.repeat_num} | "
                f"Reward: {metrics['ep_reward']:.2f} | "
                f"DetJ: {metrics['avg_detJ']:.4e} | "
                f"Rate: {metrics['sum_rate']:.2f}"
            )
            stackelberg_results.append(metrics)
        print(f"  ✓ 完成Stackelberg方法实验")
        
        # 计算统计信息
        def calculate_stats(results):
            """计算统计信息"""
            stats = {}
            for key in results[0].keys():
                if isinstance(results[0][key], (int, float)):
                    values = [r[key] for r in results]
                    stats[key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                elif isinstance(results[0][key], list) and len(results[0][key]) > 0:
                    if isinstance(results[0][key][0], (int, float)):
                        # 处理列表类型的指标（如FX, avg_tracking_error）
                        all_values = []
                        for r in results:
                            if isinstance(r[key], list):
                                all_values.extend(r[key])
                        if all_values:
                            stats[key] = {
                                "mean": float(np.mean(all_values)),
                                "std": float(np.std(all_values)),
                                "min": float(np.min(all_values)),
                                "max": float(np.max(all_values)),
                            }
                    elif isinstance(results[0][key][0], list):
                        # 处理嵌套列表（如x_auv, y_auv, tracking_error）
                        # 对于轨迹数据，计算USV的平均位置
                        if key in ["x_usv", "y_usv"]:
                            all_values = []
                            for r in results:
                                if isinstance(r[key], list):
                                    all_values.extend(r[key])
                            if all_values:
                                stats[key] = {
                                    "mean": float(np.mean(all_values)),
                                    "std": float(np.std(all_values)),
                                    "min": float(np.min(all_values)),
                                    "max": float(np.max(all_values)),
                                }
                        # 对于detJ_values，计算统计信息
                        elif key == "detJ_values":
                            all_values = []
                            for r in results:
                                if isinstance(r[key], list):
                                    all_values.extend(r[key])
                            if all_values:
                                stats[key] = {
                                    "mean": float(np.mean(all_values)),
                                    "std": float(np.std(all_values)),
                                    "min": float(np.min(all_values)),
                                    "max": float(np.max(all_values)),
                                }
                        # 对于detJ_timestamps，计算统计信息
                        elif key == "detJ_timestamps":
                            all_values = []
                            for r in results:
                                if isinstance(r[key], list):
                                    all_values.extend(r[key])
                            if all_values:
                                stats[key] = {
                                    "mean": float(np.mean(all_values)),
                                    "std": float(np.std(all_values)),
                                    "min": float(np.min(all_values)),
                                    "max": float(np.max(all_values)),
                                }
            return stats
        
        traditional_stats = calculate_stats(traditional_results)
        stackelberg_stats = calculate_stats(stackelberg_results)
        
        # 存储结果
        all_results["results"][f"delay_{delay_scenario}"] = {
            "traditional": {
                "stats": traditional_stats,
                "results": traditional_results
            },
            "stackelberg": {
                "stats": stackelberg_stats,
                "results": stackelberg_results
            }
        }
        
        # 每个场景完成后立即保存中间结果（避免数据丢失）
        try:
            serializable_partial = convert_to_native_type(all_results)
            with open(temp_result_file, 'w') as f:
                json.dump(serializable_partial, f, indent=2)
        except Exception as e:
            print(f"⚠️  保存中间结果时出错: {e}，继续运行...")
        
        # 打印对比结果
        print(f"\n丢包模式={mode_str} 的对比结果:")
        print(f"{'指标':<25} {'传统方法':<30} {'Stackelberg方法':<30} {'改进':<15}")
        print("-" * 100)
        for metric in ["ep_reward", "avg_detJ", "max_detJ", "sum_rate", "idu", "Ec", "N_DO"]:
            if metric in traditional_stats and metric in stackelberg_stats:
                trad_mean = traditional_stats[metric]["mean"]
                trad_std = traditional_stats[metric]["std"]
                stack_mean = stackelberg_stats[metric]["mean"]
                stack_std = stackelberg_stats[metric]["std"]
                
                # 计算改进百分比
                if trad_mean != 0:
                    improvement = ((stack_mean - trad_mean) / abs(trad_mean)) * 100
                    improvement_str = f"{improvement:+.2f}%"
                else:
                    improvement_str = "N/A"
                
                # detJ相关指标使用科学计数法
                if metric in ["avg_detJ", "max_detJ", "min_detJ"]:
                    trad_str = f"{trad_mean:.4e}±{trad_std:.4e}" if trad_mean != 0 else "0.0000e+00"
                    stack_str = f"{stack_mean:.4e}±{stack_std:.4e}" if stack_mean != 0 else "0.0000e+00"
                else:
                    trad_str = f"{trad_mean:.4f}±{trad_std:.4f}"
                    stack_str = f"{stack_mean:.4f}±{stack_std:.4f}"
                
                metric_name = METRIC_NAMES_CN.get(metric, metric)
                print(f"{metric_name:<25} {trad_str:<30} {stack_str:<30} {improvement_str:<15}")
    
    # 保存最终结果（使用临时文件，确保原子性）
    result_file = final_result_file
    temp_file = temp_result_file
    
    print(f"\n正在保存实验结果...")
    try:
        serializable_results = convert_to_native_type(all_results)
    except Exception as e:
        print(f"⚠️  类型转换时出错: {e}")
        print("   尝试使用更宽松的转换策略...")
        # 使用更宽松的转换策略
        def safe_convert(obj):
            try:
                return convert_to_native_type(obj)
            except:
                # 如果转换失败，尝试其他方法
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                else:
                    return str(obj)  # 最后转换为字符串
        serializable_results = {k: safe_convert(v) if not isinstance(v, (dict, list)) else 
                               ({kk: safe_convert(vv) for kk, vv in v.items()} if isinstance(v, dict) else
                                [safe_convert(vv) for vv in v]) for k, v in all_results.items()}
    
    # 先写入临时文件，然后原子性重命名（避免写入中断导致文件损坏）
    try:
        with open(temp_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)  # 使用default=str作为最后的fallback
        # 原子性重命名
        os.rename(temp_file, result_file)
        # 如果之前有临时文件，删除它
        if os.path.exists(temp_result_file) and temp_result_file != temp_file:
            os.remove(temp_result_file)
        print(f"✅ 实验结果已保存到: {result_file}")
        print(f"  目录: {result_dir}")
        
        # 同时保存pkl格式的详细数据
        pkl_file = result_file.replace('.json', '.pkl')
        try:
            with open(pkl_file, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"✅ 详细数据已保存到: {pkl_file}")
            print(f"   实验信息: {args.N_AUV}个AUV, Episode {args.load_ep}")
        except Exception as e_pkl:
            print(f"⚠️  保存pkl文件时出错: {e_pkl}")
    except Exception as e:
        # 如果出错，尝试保存到备用文件
        backup_file = f"{result_dir}/delay_comparison_{timestamp}_backup.json"
        try:
            with open(backup_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            print(f"⚠️  保存到主文件失败，已保存到备用文件: {backup_file}")
        except Exception as e2:
            print(f"❌ 保存结果时出错: {e}")
            print(f"   备用保存也失败: {e2}")
            # 如果临时文件存在，提示用户可以使用它
            if os.path.exists(temp_result_file):
                print(f"💡 中间结果已保存到: {temp_result_file}")
            raise e
    
    print(f"\n{'='*80}")
    print(f"实验结果已保存到: {result_file}")
    print(f"  目录: {result_dir}")
    print(f"  实验配置: {args.N_AUV}个AUV, Episode {args.load_ep}, {args.model_type.upper()}模型")
    print(f"{'='*80}")
    
    # 生成可视化图表（传入子文件夹路径）
    generate_delay_comparison_plots(serializable_results, result_dir, timestamp)
    
    return all_results


def generate_delay_comparison_plots(results, save_path, timestamp):
    """
    生成时延对比可视化图表（调用独立的可视化模块）
    """
    print("\n生成可视化图表...")
    try:
        # 导入可视化函数（避免循环导入和参数冲突）
        import sys
        import importlib.util
        # 保存原始sys.argv
        old_argv = sys.argv.copy()
        # 临时修改sys.argv，只保留脚本名，避免argparse解析错误
        sys.argv = [sys.argv[0]]
        
        # 使用importlib导入模块
        vis_path = os.path.join(os.path.dirname(__file__), "visualize_comparison_delay.py")
        spec = importlib.util.spec_from_file_location("visualize_comparison_delay", vis_path)
        vis_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vis_module)
        
        # 恢复原始sys.argv
        sys.argv = old_argv
        
        # 调用可视化函数
        vis_module.generate_delay_comparison_plots(results, save_path, timestamp)
        print("✅ 所有图表已生成完成！")
    except Exception as e:
        # 确保恢复sys.argv
        if 'old_argv' in locals():
            sys.argv = old_argv
        print(f"⚠️  生成图表时出错: {e}")
        import traceback
        traceback.print_exc()
        print("   你可以稍后运行: python visualize_comparison_delay.py")


if __name__ == "__main__":
    results = run_delay_comparison_experiment()
