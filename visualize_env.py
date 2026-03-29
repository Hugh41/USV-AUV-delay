"""
环境可视化脚本
实时显示USV-AUV协作环境的状态，包括：
- AUV和USV的位置和轨迹
- 传感器节点分布
- 数据收集状态
- 实时指标
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import argparse
import sys
import os
import warnings

# 配置matplotlib中文字体 (macOS)
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import Env
from td3 import TD3
import copy


def create_visualization(env, agents=None, max_steps=200, save_gif=False):
    """
    创建环境可视化
    
    Args:
        env: 环境对象
        agents: TD3智能体列表（可选，如果提供则使用智能体选择动作）
        max_steps: 最大可视化步数
        save_gif: 是否保存为GIF动画
    """
    # 初始化环境
    state = env.reset()
    
    # 设置智能体（如果提供）
    if agents is not None:
        env.set_agents(agents)
        use_agent = True
    else:
        use_agent = False
    
    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 主视图：环境地图
    ax_main = fig.add_subplot(gs[0:2, 0])
    ax_main.set_xlim(0, env.X_max)
    ax_main.set_ylim(0, env.Y_max)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('X (m)', fontsize=12)
    ax_main.set_ylabel('Y (m)', fontsize=12)
    ax_main.set_title('USV-AUV Collaborative Environment', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    
    # 指标显示区域
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_metrics.axis('off')
    ax_metrics.set_title('Real-time Metrics', fontsize=12, fontweight='bold')
    
    # 数据收集状态
    ax_data = fig.add_subplot(gs[1, 1])
    ax_data.set_title('Data Collection Status', fontsize=12, fontweight='bold')
    ax_data.set_xlabel('Sensor Node Index')
    ax_data.set_ylabel('Data Volume (Mbits)')
    ax_data.set_ylim(0, env.Fully_buffer * 1.1)
    
    # 存储轨迹
    auv_trajectories = [[] for _ in range(env.N_AUV)]
    usv_trajectory = []
    
    # 初始化绘图元素
    # 传感器节点
    sn_scatter = ax_main.scatter(
        env.SoPcenter[:, 0],
        env.SoPcenter[:, 1],
        c='gray',
        s=100,
        marker='s',
        alpha=0.6,
        label='Sensor Nodes',
        edgecolors='black',
        linewidths=0.5
    )
    
    # 目标传感器节点（高亮）
    target_scatter = ax_main.scatter(
        [],
        [],
        c='red',
        s=200,
        marker='*',
        alpha=0.8,
        label='Target Nodes',
        edgecolors='black',
        linewidths=1,
        zorder=5
    )
    
    # AUV位置和轨迹
    auv_scatters = []
    auv_lines = []
    auv_colors = plt.cm.tab10(np.linspace(0, 1, env.N_AUV))
    
    for i in range(env.N_AUV):
        line, = ax_main.plot([], [], '--', linewidth=2, 
                            color=auv_colors[i], alpha=0.5, 
                            label=f'AUV {i} Trajectory')
        auv_lines.append(line)
        scatter = ax_main.scatter([], [], s=200, c=[auv_colors[i]], 
                                 marker='o', edgecolors='black', 
                                 linewidths=2, zorder=6,
                                 label=f'AUV {i}')
        auv_scatters.append(scatter)
        
        # 数据收集半径圆圈
        circle = Circle((0, 0), env.r_dc, fill=False, 
                       linestyle='--', linewidth=1.5,
                       color=auv_colors[i], alpha=0.3)
        ax_main.add_patch(circle)
    
    # USV位置和轨迹
    usv_line, = ax_main.plot([], [], '-', linewidth=3, 
                            color='limegreen', alpha=0.7, 
                            label='USV Trajectory')
    usv_scatter = ax_main.scatter([], [], s=300, c='limegreen', 
                                  marker='^', edgecolors='black', 
                                  linewidths=2, zorder=7,
                                  label='USV')
    
    # 观测位置（带误差）
    obs_scatters = []
    for i in range(env.N_AUV):
        obs_scatter = ax_main.scatter([], [], s=100, c=[auv_colors[i]], 
                                     marker='x', linewidths=2, 
                                     alpha=0.6, zorder=5,
                                     label=f'AUV {i} Observed')
        obs_scatters.append(obs_scatter)
    
    # 数据收集状态柱状图
    bars = ax_data.bar(range(env.N_POI), [0] * env.N_POI, 
                      color='steelblue', alpha=0.7)
    ax_data.axhline(y=env.Fully_buffer, color='r', linestyle='--', 
                   linewidth=2, label='Buffer Limit')
    ax_data.legend()
    
    # 指标文本
    metrics_text = ax_metrics.text(0.1, 0.9, '', transform=ax_metrics.transAxes,
                                   fontsize=10, verticalalignment='top',
                                   family='monospace',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 初始化变量
    Ft = 0
    ep_reward = 0
    sum_rate = 0
    idu = 0
    N_DO = 0
    FX = [0] * env.N_AUV
    Ec = [0] * env.N_AUV
    mode = [0] * env.N_AUV
    hovers = [False] * env.N_AUV
    
    def animate(frame):
        nonlocal state, Ft, ep_reward, sum_rate, idu, N_DO, FX, Ec, mode, hovers
        
        if Ft >= max_steps:
            return
        
        # 选择动作
        if use_agent:
            actions = []
            for i in range(env.N_AUV):
                action = agents[i].select_action(state[i])
                actions.append(action)
        else:
            # 随机动作（用于演示）
            actions = [np.random.uniform(-1, 1, 2) for _ in range(env.N_AUV)]
        
        # 执行动作
        env.posit_change(actions, hovers)
        next_state, rewards, Done, data_rate, ec, crash = env.step_move(hovers)
        
        # 更新指标
        ep_reward += np.sum(rewards) / 1000
        N_DO += env.N_DO
        FX = np.array(FX) + np.array(env.FX)
        Ec = np.array(Ec) + np.array(ec)
        
        # 更新轨迹
        for i in range(env.N_AUV):
            auv_trajectories[i].append(env.xy[i].copy())
            if len(auv_trajectories[i]) > 1:
                traj = np.array(auv_trajectories[i])
                auv_lines[i].set_data(traj[:, 0], traj[:, 1])
            auv_scatters[i].set_offsets([env.xy[i]])
            obs_scatters[i].set_offsets([env.obs_xy[i]])
        
        usv_trajectory.append(env.usv_xy.copy())
        if len(usv_trajectory) > 1:
            traj = np.array(usv_trajectory)
            usv_line.set_data(traj[:, 0], traj[:, 1])
        usv_scatter.set_offsets([env.usv_xy])
        
        # 更新目标节点
        target_scatter.set_offsets(env.target_Pcenter)
        
        # 更新数据收集状态
        for i, bar in enumerate(bars):
            bar.set_height(env.b_S[i])
            # 根据数据量改变颜色
            if env.b_S[i] >= env.Fully_buffer:
                bar.set_color('red')
            elif env.b_S[i] > env.Fully_buffer * 0.8:
                bar.set_color('orange')
            else:
                bar.set_color('steelblue')
        
        # 更新指标文本
        metrics_str = f"""Step: {Ft}/{max_steps}
Episode Reward: {ep_reward:.2f}
Total Data Rate: {sum_rate:.2f} Mbps
Data Updates: {idu}
Overflow Rate: {N_DO/Ft if Ft > 0 else 0:.2f}
Avg Energy: {np.mean(Ec):.2f} J
Border Cross: {np.sum(FX)}
USV Position: [{env.usv_xy[0]:.1f}, {env.usv_xy[1]:.1f}]
"""
        for i in range(env.N_AUV):
            metrics_str += f"AUV {i}: [{env.xy[i][0]:.1f}, {env.xy[i][1]:.1f}]\n"
            if hovers[i]:
                metrics_str += f"  (Hovering, mode={mode[i]})\n"
        
        metrics_text.set_text(metrics_str)
        
        # 更新状态
        for i in range(env.N_AUV):
            if mode[i] == 0:
                state[i] = copy.deepcopy(next_state[i])
                if Done[i]:
                    idu += 1
                    ht = env.Q * env.updata[i] / data_rate[i] if data_rate[i] > 0 else 0
                    mode[i] = int(np.ceil(ht))
                    hovers[i] = True
                    sum_rate += data_rate[i]
            else:
                mode[i] -= 1
                if mode[i] == 0:
                    hovers[i] = False
                    state[i] = env.CHOOSE_AIM(idx=i, lamda=0.05)
        
        Ft += 1
        env.Ft = Ft
        
        return (auv_scatters + obs_scatters + [usv_scatter, target_scatter] + 
                auv_lines + [usv_line] + list(bars) + [metrics_text])
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=max_steps, 
                                  interval=100, blit=False, repeat=False)
    
    # 添加图例
    ax_main.legend(loc='upper right', fontsize=9)
    
    if save_gif:
        print("Saving GIF animation...")
        anim.save('env_visualization.gif', writer='pillow', fps=10)
        print("GIF saved as env_visualization.gif")
    else:
        plt.show()
    
    return anim


def main():
    parser = argparse.ArgumentParser(description='Environment Visualization')
    parser.add_argument('--max_steps', type=int, default=200, 
                       help='Maximum visualization steps (default: 200)')
    parser.add_argument('--load_ep', type=int, default=None,
                       help='Load trained model episode number (optional)')
    parser.add_argument('--save_gif', action='store_true',
                       help='Save as GIF animation')
    parser.add_argument('--N_AUV', type=int, default=2, help='Number of AUVs')
    parser.add_argument('--n_s', type=int, default=30, help='Number of sensor nodes')
    parser.add_argument('--border_x', type=float, default=200.0, help='X border')
    parser.add_argument('--border_y', type=float, default=200.0, help='Y border')
    parser.add_argument('--episode_length', type=int, default=1000, help='Episode length')
    parser.add_argument('--R_dc', type=float, default=6.0, help='Data collection radius')
    
    args = parser.parse_args()
    
    # 创建环境
    env = Env(args)
    
    # 加载模型（如果指定）
    agents = None
    if args.load_ep is not None:
        try:
            from td3 import TD3
            import torch
            state_dim = env.state_dim
            action_dim = 2
            agents = [TD3(state_dim, action_dim) for _ in range(args.N_AUV)]
            
            for i, agent in enumerate(agents):
                # 检查模型文件是否存在（根据AUV数量动态设置路径）
                model_path = f"models_ddpg_{args.N_AUV}/"
                actor_path = f"{model_path}TD3_{i}_{args.load_ep}_actor.pth"
                critic_path = f"{model_path}TD3_{i}_{args.load_ep}_critic.pth"
                if os.path.exists(actor_path) and os.path.exists(critic_path):
                    # 使用TD3的load方法（需要目录路径，不带文件名）
                    try:
                        agent.load(model_path, args.load_ep, idx=i)
                        print(f"✓ Loaded AUV {i} model (Episode {args.load_ep})")
                    except Exception as e:
                        print(f"⚠ Failed to load AUV {i} model: {e}")
                        agents = None
                        break
                else:
                    print(f"⚠ Model file not found: {actor_path}")
                    agents = None
                    break
        except Exception as e:
            print(f"⚠ Failed to load models: {e}")
            print("Using random actions for visualization")
            agents = None
    
    # 创建可视化
    print("Starting environment visualization...")
    print("Tip: Close window to stop visualization")
    anim = create_visualization(env, agents, args.max_steps, args.save_gif)
    
    if not args.save_gif:
        plt.show()


if __name__ == '__main__':
    main()

