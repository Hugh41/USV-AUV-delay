"""
Stackelberg博弈 vs 传统方法对比实验

该脚本用于对比Stackelberg博弈方法和传统优化方法的效果
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

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 指标中文名称映射 (仅保留核心关注指标)
METRIC_NAMES_CN = {
    "avg_detJ": "平均Fisher信息矩阵detJ",
    "sum_rate": "总数据率",
    "Ec": "平均能耗",
    "idu": "数据更新次数",
    "N_DO": "数据溢出率",
}

# 参数设置
parser = argparse.ArgumentParser()
# ------ 实验参数 ------
parser.add_argument("--repeat_num", type=int, default=10, help="重复实验次数")
parser.add_argument(
    "--episode_length", type=int, default=1000, help="每个episode的长度 (sec)"
)
parser.add_argument("--load_ep", type=int, default=575, help="加载的模型episode")
parser.add_argument("--use_stackelberg", type=int, default=1, help="是否使用Stackelberg博弈 (1=是, 0=否)")
parser.add_argument("--skip_model_check", action="store_true", help="跳过模型检查，使用随机初始化的模型（仅用于测试）")
parser.add_argument("--model_type", type=str, default="td3", choices=["td3", "dsac"],
                    help="模型类型: td3 (models_td3) 或 dsac (models_dsac)")
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
parser.add_argument(
    "--skip_plots", action="store_true",
    help="跳过生成图表（仅保存数据）"
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
RES_PATH = BASE_PATH + "/comparison_results"
if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)


def calculate_detJ(env):
    """
    计算当前时刻的Fisher信息矩阵detJ值
    """
    try:
        # 计算负的detJ，然后取负得到正的detJ
        neg_detJ = env.calcnegdetJ_USV(env.usv_xy)
        # calcnegdetJ_USV返回的是负的detJ值，所以需要取负
        detJ = -neg_detJ
        # 确保detJ非负
        detJ = max(0, detJ)
        return detJ
    except Exception as e:
        print(f"计算detJ时出错: {e}")
        return 0


def run_single_episode(env, agents, use_stackelberg=True):
    """
    运行单个episode并记录所有指标
    
    Args:
        env: 环境对象
        agents: AUV智能体列表
        use_stackelberg: 是否使用Stackelberg博弈
        
    Returns:
        metrics: 包含所有指标的字典
    """
    # 设置是否使用Stackelberg博弈
    env.use_stackelberg = use_stackelberg
    
    state_c = env.reset()
    state = copy.deepcopy(state_c)
    
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
    
    # Fisher信息矩阵detJ值记录
    detJ_values = [calculate_detJ(env)]
    detJ_timestamps = [0]
    
    # USV位置变化记录
    usv_position_changes = [0]  # 初始位置变化为0
    
    while True:
        act = []
        for i in range(env.N_AUV):
            iact = agents[i].select_action(state[i])
            act.append(iact)
        
        # 记录USV位置变化
        prev_usv_xy = copy.deepcopy(env.usv_xy)
        
        env.posit_change(act, hovers)
        state_, rewards, Done, data_rate, ec, cs = env.step_move(hovers)
        
        # 计算USV位置变化距离
        usv_move_distance = np.linalg.norm(env.usv_xy - prev_usv_xy)
        usv_position_changes.append(usv_move_distance)
        
        # 记录detJ值
        detJ = calculate_detJ(env)
        detJ_values.append(detJ)
        detJ_timestamps.append(Ft + 1)
        
        crash += cs
        ep_reward += np.sum(rewards) / 1000
        
        for i in range(env.N_AUV):
            # 记录轨迹
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
        
        Ft += 1
        env.Ft = Ft
        N_DO += env.N_DO
        FX = np.array(FX) + np.array(env.FX)
        DQ += sum(env.b_S / env.Fully_buffer)
        Ec = np.array(Ec) + np.array(ec)
        
        if Ft > args.episode_length:
            # 计算平均指标
            N_DO /= Ft
            DQ /= Ft
            DQ /= env.N_POI
            Ec = np.sum(np.array(Ec) / (Ft - np.array(Ht))) / env.N_AUV
            
            # 计算平均detJ值
            avg_detJ = np.mean(detJ_values) if len(detJ_values) > 0 else 0
            max_detJ = np.max(detJ_values) if len(detJ_values) > 0 else 0
            min_detJ = np.min(detJ_values) if len(detJ_values) > 0 else 0
            
            # 计算USV平均移动距离
            avg_usv_move = np.mean(usv_position_changes) if len(usv_position_changes) > 0 else 0
            total_usv_move = np.sum(usv_position_changes)
            
            # 计算平均跟踪误差
            avg_tracking_error = [
                np.mean(tracking_error[i]) for i in range(env.N_AUV)
            ]
            
            metrics = {
                "ep_reward": ep_reward,
                "DQ": DQ,
                "sum_rate": sum_rate,
                "idu": idu,
                "Ec": Ec,
                "N_DO": N_DO,
                "crash": crash,
                "FX": FX.tolist() if isinstance(FX, np.ndarray) else FX,
                "avg_detJ": avg_detJ,
                "max_detJ": max_detJ,
                "min_detJ": min_detJ,
                "avg_usv_move": avg_usv_move,
                "total_usv_move": total_usv_move,
                "avg_tracking_error": avg_tracking_error,
                "detJ_values": detJ_values,
                "detJ_timestamps": detJ_timestamps,
                "x_auv": x_auv,
                "y_auv": y_auv,
                "x_usv": x_usv,
                "y_usv": y_usv,
                "tracking_error": tracking_error,
                # 环境静态信息（用于可视化）
                "SoPcenter": env.SoPcenter.tolist() if isinstance(env.SoPcenter, np.ndarray) else env.SoPcenter,
                "lda": env.lda,
            }
            break
    
    return metrics


def check_model_files(save_path, load_ep, n_auv, model_type="td3"):
    """
    检查模型文件是否存在
    
    Args:
        save_path: 模型保存路径
        load_ep: 要加载的episode
        n_auv: AUV数量
        model_type: 模型类型 ("td3" 或 "dsac")
    
    Returns:
        bool: 如果所有模型文件都存在返回True，否则返回False
        list: 缺失的文件列表
    """
    missing_files = []
    for i in range(n_auv):
        if model_type == "dsac":
            # DSAC模型文件格式: DSAC_{idx}_{ep}.pkl
            files_to_check = [
                f"{save_path}DSAC_{i}_{load_ep}.pkl",
            ]
        else:
            # TD3模型文件格式
            ep = "_" + str(load_ep)
            idx = "_" + str(i)
            files_to_check = [
                f"{save_path}TD3{idx}{ep}_critic.pth",
                f"{save_path}TD3{idx}{ep}_critic_optimizer.pth",
                f"{save_path}TD3{idx}{ep}_actor.pth",
                f"{save_path}TD3{idx}{ep}_actor_optimizer.pth",
            ]
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files


def run_comparison_experiment():
    """
    运行对比实验
    """
    print("=" * 80)
    print("Stackelberg博弈 vs 传统方法对比实验")
    print(f"模型类型: {args.model_type.upper()}")
    print("=" * 80)
    
    # 检查模型文件
    model_exists, missing_files = check_model_files(SAVE_PATH, args.load_ep, args.N_AUV, args.model_type)
    
    if not model_exists and not args.skip_model_check:
        print("\n❌ 错误: 模型文件不存在!")
        print(f"模型路径: {SAVE_PATH}")
        print(f"Episode: {args.load_ep}")
        print(f"AUV数量: {args.N_AUV}")
        print("\n缺失的文件:")
        for f in missing_files[:10]:  # 只显示前10个
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... 还有 {len(missing_files) - 10} 个文件缺失")
        
        print("\n💡 解决方案:")
        print("1. 先训练模型:")
        print("   python train_td3.py")
        print("\n2. 或者使用已存在的模型episode编号:")
        print("   python compare_stackelberg.py --load_ep <episode_number>")
        print("\n3. 或者检查模型文件路径是否正确")
        
        # 尝试查找可用的模型文件
        if os.path.exists(SAVE_PATH):
            print(f"\n📁 检查 {SAVE_PATH} 目录中的文件...")
            try:
                files = os.listdir(SAVE_PATH)
                pth_files = [f for f in files if f.endswith('.pth')]
                if pth_files:
                    print(f"找到 {len(pth_files)} 个.pth文件:")
                    # 尝试提取episode编号
                    episodes = set()
                    for f in pth_files[:10]:
                        print(f"  - {f}")
                        # 尝试从文件名提取episode
                        parts = f.split('_')
                        for part in parts:
                            if part.isdigit() and len(part) >= 3:
                                episodes.add(int(part))
                    if episodes:
                        print(f"\n可用的episode编号: {sorted(episodes)}")
                        print(f"尝试使用: python compare_stackelberg.py --load_ep {max(episodes)}")
                else:
                    print("  未找到.pth文件")
            except Exception as e:
                print(f"  无法读取目录: {e}")
        else:
            print(f"\n目录 {SAVE_PATH} 不存在")
            print("请先运行训练脚本创建模型文件")
        
        return None
    
    # 创建环境
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
    
    if args.skip_model_check:
        print("\n⚠️  警告: 使用随机初始化的模型（未训练）")
        print("   这仅用于测试环境，结果可能不准确")
    else:
        print(f"\n📦 加载{args.model_type.upper()}模型文件 (Episode {args.load_ep})...")
        try:
            for i in range(N_AUV):
                agents[i].load(SAVE_PATH, args.load_ep, idx=i)
            print("✅ 模型加载成功!")
        except Exception as e:
            print(f"❌ 加载模型时出错: {e}")
            return None
    
    # 设置agents到环境中
    env.set_agents(agents)
    
    # 存储结果
    traditional_results = []
    stackelberg_results = []
    
    # 运行传统方法实验
    print("\n" + "=" * 80)
    print("运行传统方法实验...")
    print("=" * 80)
    env.use_stackelberg = False
    
    start_time = time.time()
    for ep in range(args.repeat_num):
        print(f"\n传统方法 - Episode {ep + 1}/{args.repeat_num}")
        metrics = run_single_episode(env, agents, use_stackelberg=False)
        traditional_results.append(metrics)
        print(
            f"  Reward: {metrics['ep_reward']:.2f} | "
            f"Avg detJ: {metrics['avg_detJ']:.6f} | "
            f"Data Rate: {metrics['sum_rate']:.2f} | "
            f"Energy: {metrics['Ec']:.2f}"
        )
    traditional_time = time.time() - start_time
    
    # 运行Stackelberg方法实验
    print("\n" + "=" * 80)
    print("运行Stackelberg博弈方法实验...")
    print("=" * 80)
    env.use_stackelberg = True
    
    start_time = time.time()
    for ep in range(args.repeat_num):
        print(f"\nStackelberg方法 - Episode {ep + 1}/{args.repeat_num}")
        metrics = run_single_episode(env, agents, use_stackelberg=True)
        stackelberg_results.append(metrics)
        print(
            f"  Reward: {metrics['ep_reward']:.2f} | "
            f"Avg detJ: {metrics['avg_detJ']:.6f} | "
            f"Data Rate: {metrics['sum_rate']:.2f} | "
            f"Energy: {metrics['Ec']:.2f}"
        )
    stackelberg_time = time.time() - start_time
    
    # 计算统计结果
    def convert_to_native_type(obj):
        """将NumPy类型转换为Python原生类型，以便JSON序列化"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_type(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_native_type(item) for item in obj)
        else:
            return obj

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
                    # 处理列表类型的指标（如FX）
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
    
    # 打印对比结果
    print("\n" + "=" * 80)
    print("对比实验结果")
    print("=" * 80)
    
    print("\n关键指标对比:")
    print("-" * 80)
    print(f"{'指标':<20} {'传统方法':<25} {'Stackelberg方法':<25} {'改进':<15}")
    print("-" * 80)
    
    key_metrics = [
        "ep_reward",
        "avg_detJ",
        "max_detJ",
        "sum_rate",
        "idu",
        "Ec",
        "N_DO",
        "avg_tracking_error",
        "avg_usv_move",
    ]
    
    for metric in key_metrics:
        if metric in traditional_stats and metric in stackelberg_stats:
            trad_mean = traditional_stats[metric]["mean"]
            stack_mean = stackelberg_stats[metric]["mean"]
            
            if trad_mean != 0:
                improvement = ((stack_mean - trad_mean) / abs(trad_mean)) * 100
            else:
                improvement = 0
            
            # 处理列表类型的指标
            if metric == "avg_tracking_error":
                trad_str = f"{trad_mean:.4f}±{traditional_stats[metric]['std']:.4f}"
                stack_str = f"{stack_mean:.4f}±{stackelberg_stats[metric]['std']:.4f}"
            else:
                trad_str = f"{trad_mean:.4f}±{traditional_stats[metric]['std']:.4f}"
                stack_str = f"{stack_mean:.4f}±{stackelberg_stats[metric]['std']:.4f}"
            
            improvement_str = f"{improvement:+.2f}%"
            print(f"{metric:<20} {trad_str:<25} {stack_str:<25} {improvement_str:<15}")
    
    print("\n计算时间对比:")
    print(f"  传统方法: {traditional_time:.2f}秒 ({traditional_time/args.repeat_num:.2f}秒/episode)")
    print(f"  Stackelberg方法: {stackelberg_time:.2f}秒 ({stackelberg_time/args.repeat_num:.2f}秒/episode)")
    print(f"  时间增加: {((stackelberg_time - traditional_time) / traditional_time * 100):+.2f}%")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    usv_freq = args.usv_update_frequency
    model_type = args.model_type
    
    # 创建以时间戳命名的子文件夹，后缀为 usv_update_frequency 和 model_type
    result_dir = f"{RES_PATH}/comparison_{timestamp}_{usv_freq}_{model_type}"
    os.makedirs(result_dir, exist_ok=True)
    
    result_file = f"{result_dir}/comparison_{timestamp}.json"
    
    comparison_data = {
        "experiment_info": {
            "timestamp": timestamp,
            "repeat_num": args.repeat_num,
            "episode_length": args.episode_length,
            "N_AUV": N_AUV,
            "n_s": args.n_s,
            "usv_update_frequency": args.usv_update_frequency,
            "border_x": args.border_x,
            "border_y": args.border_y,
            "model_type": args.model_type,
            "load_ep": args.load_ep,
            # H is hardcoded in Env as 100
            "H": 100, 
        },
        "traditional": {
            "stats": traditional_stats,
            "time": traditional_time,
            "results": traditional_results,
        },
        "stackelberg": {
            "stats": stackelberg_stats,
            "time": stackelberg_time,
            "results": stackelberg_results,
        },
    }
    
    # 保存为JSON（简化数据）
    simplified_data = {
        "experiment_info": comparison_data["experiment_info"],
        "traditional": {
            "stats": convert_to_native_type(traditional_stats),
            "time": float(traditional_time),
        },
        "stackelberg": {
            "stats": convert_to_native_type(stackelberg_stats),
            "time": float(stackelberg_time),
        },
    }
    
    with open(result_file, "w") as f:
        json.dump(simplified_data, f, indent=2)
    
    # 保存完整数据为pickle
    pickle_file = f"{result_dir}/comparison_{timestamp}.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(comparison_data, f)
    
    print(f"\n结果已保存到:")
    print(f"  目录: {result_dir}")
    print(f"  JSON: {result_file}")
    print(f"  Pickle: {pickle_file}")
    
    # 自动生成可视化图表（除非用户指定跳过）
    if not args.skip_plots:
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
            vis_path = os.path.join(os.path.dirname(__file__), "visualize_comparison.py")
            spec = importlib.util.spec_from_file_location("visualize_comparison", vis_path)
            vis_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vis_module)
            
            # 恢复原始sys.argv
            sys.argv = old_argv
            
            # 调用可视化函数（传入时间戳和结果目录）
            # 1. 核心统计图表
            vis_module.plot_detJ_evolution(comparison_data, result_dir, timestamp)
            vis_module.plot_summary_table(comparison_data, result_dir, timestamp)
            
            # 2. 高级可视化图表 (轨迹与误差演化)
            exp_info = comparison_data.get("experiment_info", {})
            
            # 绘制传统方法
            if "results" in comparison_data["traditional"] and comparison_data["traditional"]["results"]:
                vis_module.plot_trajectory(comparison_data["traditional"]["results"], "Traditional", result_dir, timestamp)
                vis_module.plot_tracking_error_evolution(comparison_data["traditional"]["results"], exp_info, "Traditional", result_dir, timestamp)
                
            # 绘制Stackelberg方法
            if "results" in comparison_data["stackelberg"] and comparison_data["stackelberg"]["results"]:
                vis_module.plot_trajectory(comparison_data["stackelberg"]["results"], "Stackelberg", result_dir, timestamp)
                vis_module.plot_tracking_error_evolution(comparison_data["stackelberg"]["results"], exp_info, "Stackelberg", result_dir, timestamp)

            print("✅ 所有图表已生成完成！")
        except Exception as e:
            # 确保恢复sys.argv
            if 'old_argv' in locals():
                sys.argv = old_argv
            print(f"⚠️  生成图表时出错: {e}")
            import traceback
            traceback.print_exc()
            print("   你可以稍后运行: python visualize_comparison.py")
    
    return comparison_data


if __name__ == "__main__":
    comparison_data = run_comparison_experiment()
    if comparison_data is not None:
        print("\n✅ 实验完成！")
    else:
        print("\n❌ 实验失败，请检查错误信息并修复问题后重试。")
        exit(1)

