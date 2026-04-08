"""
使用DSAC-T算法训练USV-AUV环境
每个AUV有独立的DSAC网络，类似TD3的训练方式
"""
import os
import sys
import argparse
import numpy as np
import copy
import time

# 添加DSAC-v2路径
dsac_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DSAC-v2')
sys.path.insert(0, dsac_path)

import torch


class DSACAgent:
    """单个AUV的DSAC Agent"""
    
    def __init__(self, agent_idx, **kwargs):
        self.agent_idx = agent_idx
        
        # 添加ReplayBuffer需要的参数
        kwargs["trainer"] = "off_serial_trainer"
        kwargs["additional_info"] = {}
        
        self.kwargs = kwargs
        
        # 导入DSAC模块
        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        module = __import__(alg_file_name)
        self.ApproxContainer = getattr(module, "ApproxContainer")
        alg_cls = getattr(module, alg_name)
        
        # 创建网络
        self.networks = self.ApproxContainer(**kwargs)
        
        # 创建算法
        self.alg = alg_cls(**kwargs)
        self.alg.networks = self.networks
        
        # 创建缓冲区
        from training.replay_buffer import ReplayBuffer
        self.buffer = ReplayBuffer(index=agent_idx, **kwargs)
        
        # 训练参数
        self.replay_batch_size = kwargs["replay_batch_size"]
        self.use_gpu = kwargs.get("use_gpu", False)
        
        if self.use_gpu:
            self.networks.cuda()
    
    def select_action(self, obs, deterministic=False):
        """选择动作"""
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
    
    def store_transition(self, obs, action, reward, next_obs, done, logp=0.0, info=None):
        """存储经验"""
        if info is None:
            info = {}
        data = [(obs.copy(), info, action, reward, next_obs.copy(), done, logp, info)]
        self.buffer.add_batch(data)
    
    def train_step(self, iteration):
        """训练一步"""
        if self.buffer.size < self.kwargs["buffer_warm_size"]:
            return None
        
        # 采样
        replay_samples = self.buffer.sample_batch(self.replay_batch_size)
        
        if self.use_gpu:
            for k, v in replay_samples.items():
                replay_samples[k] = v.cuda()
        
        # 训练
        alg_tb_dict = self.alg.local_update(replay_samples, iteration)
        
        return alg_tb_dict
    
    def save(self, filepath):
        """保存模型"""
        torch.save(self.networks.state_dict(), filepath)
        
    def load(self, filepath):
        """加载模型"""
        self.networks.load_state_dict(torch.load(filepath))


def create_multi_env(**kwargs):
    """创建多智能体环境"""
    env_gym_path = os.path.join(dsac_path, "env_gym")
    sys.path.insert(0, env_gym_path)
    
    from gym_usv_auv_multi_data import USVAUVMultiEnv
    return USVAUVMultiEnv(**kwargs)


def init_args_for_agent(env, **kwargs):
    """为单个agent初始化参数"""
    from utils.common_utils import seed_everything
    
    # 设置观测和动作维度
    kwargs["obsv_dim"] = env.observation_space.shape[0]
    kwargs["action_dim"] = env.action_space.shape[0]
    kwargs["action_high_limit"] = env.action_space.high.astype('float32')
    kwargs["action_low_limit"] = env.action_space.low.astype('float32')
    
    # 设置其他必要参数
    kwargs["batch_size_per_sampler"] = kwargs.get("sample_batch_size", 20)
    kwargs["use_gpu"] = False
    if kwargs.get("enable_cuda", False) and torch.cuda.is_available():
        kwargs["use_gpu"] = True
    
    # 设置随机种子
    seed = kwargs.get("seed", None)
    kwargs["seed"] = seed_everything(seed) if seed else np.random.randint(0, 10000)
    
    # 设置CNN相关（如果需要）
    kwargs["cnn_shared"] = False
    
    return kwargs


if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser()

    ################################################
    # 关键参数
    parser.add_argument("--env_id", type=str, default="gym_usv_auv_multi", help="环境ID")
    parser.add_argument("--algorithm", type=str, default="DSAC_V2", help="DSAC_V2 或 DSAC_V1")
    parser.add_argument("--enable_cuda", default=False, help="启用CUDA")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id to use (-1 = auto)")
    parser.add_argument("--seed", default=None, help="随机种子")

    ################################################
    # 1. 环境参数
    parser.add_argument("--reward_scale", type=float, default=1.0, help="奖励缩放因子")
    parser.add_argument("--action_type", type=str, default="continu", help="动作类型: continu/discret")
    
    # USV-AUV特定环境参数
    parser.add_argument("--R_dc", type=float, default=6.0, help="数据收集半径")
    parser.add_argument("--border_x", type=float, default=200.0, help="区域x大小")
    parser.add_argument("--border_y", type=float, default=200.0, help="区域y大小")
    parser.add_argument("--n_s", type=int, default=30, help="SNs数量")
    parser.add_argument("--N_AUV", type=int, default=2, help="AUV数量")
    parser.add_argument("--Q", type=float, default=2.0, help="SNs容量")
    parser.add_argument("--alpha", type=float, default=0.05, help="SNs选择距离优先级")
    parser.add_argument("--episode_length", type=int, default=1000, help="Episode长度")
    parser.add_argument(
        "--usv_update_frequency", type=int, default=5,
        help="USV更新频率（每N个时间步更新一次，训练和推理必须一致）"
    )

    ################################################
    # 2.1 价值函数近似参数
    parser.add_argument("--value_func_name", type=str, default="ActionValueDistri",
                        help="价值函数名称: StateValue/ActionValue/ActionValueDis/ActionValueDistri")
    parser.add_argument("--value_func_type", type=str, default="MLP", 
                        help="价值函数类型: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256, 256])
    parser.add_argument("--value_hidden_activation", type=str, default="gelu", 
                        help="激活函数: relu/gelu/elu/selu/sigmoid/tanh")
    parser.add_argument("--value_output_activation", type=str, default="linear", help="输出激活: linear/tanh")
    parser.add_argument("--value_min_log_std", type=int, default=-8)
    parser.add_argument("--value_max_log_std", type=int, default=8)

    # 2.2 策略函数近似参数
    parser.add_argument("--policy_func_name", type=str, default="StochaPolicy",
                        help="策略函数名称: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy")
    parser.add_argument("--policy_func_type", type=str, default="MLP", 
                        help="策略函数类型: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    parser.add_argument("--policy_act_distribution", type=str, default="TanhGaussDistribution",
                        help="动作分布: default/TanhGaussDistribution/GaussDistribution")
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256, 256])
    parser.add_argument("--policy_hidden_activation", type=str, default="gelu", 
                        help="激活函数: relu/gelu/elu/selu/sigmoid/tanh")
    parser.add_argument("--policy_output_activation", type=str, default="linear", help="输出激活: linear/tanh")
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=0.5)

    ################################################
    # 3. RL算法参数
    parser.add_argument("--value_learning_rate", type=float, default=0.0001)
    parser.add_argument("--policy_learning_rate", type=float, default=0.0001)
    parser.add_argument("--alpha_learning_rate", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.97, help="折扣因子（与TD3保持一致）")
    parser.add_argument("--tau", type=float, default=0.005, help="软更新系数")
    parser.add_argument("--auto_alpha", type=bool, default=True, help="自动调整熵系数")
    parser.add_argument("--entropy_alpha", type=float, default=0.2, help="熵系数（如果auto_alpha=False）")
    parser.add_argument("--delay_update", type=int, default=2, help="延迟更新频率")
    parser.add_argument("--TD_bound", type=float, default=1, help="TD误差边界")
    parser.add_argument("--bound", default=True, help="是否使用边界")

    ################################################
    # 4. 缓冲区参数
    parser.add_argument("--buffer_name", type=str, default="replay_buffer", 
                        help="缓冲区类型: replay_buffer/prioritized_replay_buffer")
    parser.add_argument("--buffer_warm_size", type=int, default=1000, help="缓冲区预热大小")
    parser.add_argument("--buffer_max_size", type=int, default=500000, help="缓冲区最大大小")
    parser.add_argument("--replay_batch_size", type=int, default=256, help="回放批次大小")
    parser.add_argument("--sample_batch_size", type=int, default=20, help="采样批次大小")

    ################################################
    # 5. 训练参数
    parser.add_argument("--episode_num", type=int, default=600, help="训练episode数量（与TD3保持一致，默认600）")
    parser.add_argument("--save_model_freq", type=int, default=25, help="每N个episode保存一次模型")
    parser.add_argument("--models_dir", type=str, default="models_dsac", help="模型保存目录")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔（episode）")
    
    # 继续训练参数
    parser.add_argument("--load_ep", type=int, default=None, help="加载的episode编号（用于继续训练）")
    parser.add_argument("--start_episode", type=int, default=0, help="起始episode（用于继续训练）")

    # 提前停止参数（默认关闭，与TD3对比时保持一致）
    parser.add_argument("--early_stop", type=bool, default=False, help="是否启用提前停止（默认False，与TD3对比时保持一致）")
    parser.add_argument("--early_stop_patience", type=int, default=50, help="提前停止耐心值（连续N个episode未提升）")
    parser.add_argument("--early_stop_threshold", type=float, default=1.0, help="提前停止阈值")
    parser.add_argument("--early_stop_min_episodes", type=int, default=200, help="最小训练episodes")

    ################################################
    # 获取参数字典
    args = vars(parser.parse_args())

    # Set GPU device
    gpu_id = args.pop("gpu", -1)
    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        args["use_gpu"] = True
        args["enable_cuda"] = True

    # 设置熵系数
    entropy_alpha = args.pop("entropy_alpha", 0.2)
    args["alpha"] = entropy_alpha
    
    # 创建模型保存目录
    models_dir = args["models_dir"]
    # 如果是默认目录，添加AUV数量和更新频率后缀：models_dsac_2AUV_5
    if models_dir == "models_dsac":
        models_dir = f"{models_dir}_{args['N_AUV']}AUV_{args['usv_update_frequency']}"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"创建模型保存目录: {models_dir}")

    ################################################
    # 创建环境
    print("=" * 80)
    print("创建多智能体环境...")
    env = create_multi_env(**args)
    N_AUV = env.n_agents
    print(f"环境创建成功！AUV数量: {N_AUV}")
    
    # 初始化参数
    args = init_args_for_agent(env, **args)
    
    ################################################
    # 为每个AUV创建独立的DSAC Agent
    print("\n创建DSAC Agents（每个AUV独立网络）...")
    agents = []
    for i in range(N_AUV):
        agent_args = copy.deepcopy(args)
        agent_args["seed"] = args["seed"] + i * 100 if args["seed"] else None
        agent = DSACAgent(i, **agent_args)
        agents.append(agent)
        print(f"  ✓ 创建AUV{i}的DSAC Agent")
    
    # 加载模型（如果指定）
    if args.get("load_ep") is not None:
        load_ep = args["load_ep"]
        print(f"\n加载Episode {load_ep}的模型...")
        for i in range(N_AUV):
            model_path = os.path.join(models_dir, f"DSAC_{i}_{load_ep}.pkl")
            if os.path.exists(model_path):
                agents[i].load(model_path)
                print(f"  ✓ 加载AUV{i}模型: {model_path}")
            else:
                print(f"  ✗ 模型不存在: {model_path}")
    
    ################################################
    # 设置Stackelberg博弈
    print("\n设置Stackelberg博弈...")
    env.env.use_stackelberg = True
    env.set_agents(agents)
    print("  ✓ Stackelberg博弈已启用")
    print(f"  ✓ USV更新频率: {env.env.usv_update_frequency}")
    
    ################################################
    # 开始训练
    print("\n" + "=" * 80)
    print("开始DSAC-T训练（多AUV独立网络）")
    print("=" * 80)
    print(f"算法: {args['algorithm']}")
    print(f"AUV数量: {N_AUV}")
    print(f"Episode数量: {args['episode_num']}")
    print(f"Episode长度: {args['episode_length']}")
    print(f"模型保存目录: {models_dir}")
    print(f"网络架构: 每个AUV独立网络（类似TD3）")
    print(f"Stackelberg博弈: 已启用")
    print(f"USV更新频率: {env.env.usv_update_frequency}")
    print("=" * 80)
    
    # 训练参数
    episode_num = args["episode_num"]
    episode_length = args["episode_length"]
    save_model_freq = args["save_model_freq"]
    log_interval = args["log_interval"]
    start_episode = args.get("start_episode", 0)
    
    # 提前停止相关
    early_stop_enabled = args.get("early_stop", False)
    early_stop_patience = args.get("early_stop_patience", 50)
    early_stop_threshold = args.get("early_stop_threshold", 1.0)
    early_stop_min_episodes = args.get("early_stop_min_episodes", 200)
    
    best_ep_reward = float('-inf')
    patience_counter = 0
    ep_reward_history = []
    
    if early_stop_enabled:
        print(f"\n提前停止已启用:")
        print(f"  耐心值: {early_stop_patience} episodes")
        print(f"  提升阈值: {early_stop_threshold}")
        print(f"  最小训练episodes: {early_stop_min_episodes}")
    
    # 噪声参数
    noise = 0.5
    
    # 打印表头
    print(f"\n{'='*100}")
    header = f"{'Episode':<10}"
    for i in range(N_AUV):
        header += f" {'AUV'+str(i)+'_Reward':<12}"
    header += f" {'Total':<12} {'Avg(100)':<12} {'Data_Rate':<12} {'Noise':<8} {'Progress':<10}"
    print(header)
    print(f"{'='*100}")
    
    # 训练循环
    total_steps = 0
    start_time = time.time()
    
    for ep in range(start_episode, episode_num):
        # 重置环境
        all_states = env.reset()
        states = [all_states[i].copy().astype(np.float32) for i in range(N_AUV)]
        
        ep_rewards = [0.0] * N_AUV
        ep_data_rate = 0.0
        step = 0
        
        while step < episode_length:
            # 每个AUV选择动作
            actions = []
            for i in range(N_AUV):
                action = agents[i].select_action(states[i], deterministic=False)
                # 添加探索噪声
                action = np.clip(action + noise * np.random.randn(2), -1, 1)
                actions.append(action)
            
            # 执行动作
            next_states, rewards, dones, infos = env.step(actions)
            
            # 存储经验并训练每个agent
            for i in range(N_AUV):
                # 存储经验
                agents[i].store_transition(
                    states[i], actions[i], rewards[i], 
                    next_states[i].astype(np.float32), dones[i]
                )
                
                # 训练
                if agents[i].buffer.size >= args["buffer_warm_size"]:
                    agents[i].train_step(total_steps)
                
                # 更新状态
                states[i] = next_states[i].copy().astype(np.float32)
                ep_rewards[i] += rewards[i]
                
                # 统计数据率
                ep_data_rate += infos[i].get('data_rate', 0)
            
            step += 1
            total_steps += 1
            
            # 衰减噪声
            noise = max(noise * 0.99998, 0.1)
            
            # 检查是否结束
            if all(dones):
                break
        
        # 记录episode信息
        total_reward = sum(ep_rewards)
        ep_reward_history.append(total_reward)
        avg_reward = np.mean(ep_reward_history[-100:])
        
        # 保存收敛数据到文件
        with open(os.path.join(models_dir, "training_log.txt"), "a") as f:
            if ep == start_episode and not os.path.exists(os.path.join(models_dir, "training_log.txt")):
                f.write("Episode,Total Reward,Avg Reward (100),Data Rate,Noise\n")
            f.write(f"{ep},{total_reward},{avg_reward},{ep_data_rate},{noise}\n")
        
        # 打印日志
        if ep % log_interval == 0 or ep == episode_num - 1:
            progress = (ep + 1) / episode_num * 100
            log_str = f"{ep:<10}"
            for i in range(N_AUV):
                log_str += f" {ep_rewards[i]:<12.2f}"
            log_str += f" {total_reward:<12.2f} {avg_reward:<12.2f} {ep_data_rate:<12.2f} {noise:<8.4f} {progress:>6.2f}%"
            print(log_str)
        
        # 保存模型
        if ep % save_model_freq == 0 and ep > 0:
            print(f"\n[Episode {ep}] 保存模型...")
            for i in range(N_AUV):
                model_path = os.path.join(models_dir, f"DSAC_{i}_{ep}.pkl")
                agents[i].save(model_path)
                print(f"  ✓ 保存AUV{i}模型: {model_path}")
            print(f"{'='*100}")
        
        # 提前停止检查
        if early_stop_enabled and ep >= early_stop_min_episodes:
            window_size = min(10, len(ep_reward_history))
            recent_avg_reward = np.mean(ep_reward_history[-window_size:])
            
            if recent_avg_reward > best_ep_reward + early_stop_threshold:
                improvement = recent_avg_reward - best_ep_reward
                best_ep_reward = recent_avg_reward
                patience_counter = 0
                if ep % 10 == 0:
                    print(f"  ✓ 性能提升: {improvement:.2f}, 最佳平均奖励: {best_ep_reward:.2f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\n{'='*80}")
                    print(f"提前停止触发！")
                    print(f"  Episode: {ep}/{episode_num}")
                    print(f"  最佳平均奖励: {best_ep_reward:.2f}")
                    print(f"  当前平均奖励: {recent_avg_reward:.2f}")
                    print(f"  连续{patience_counter}个episode未显著提升")
                    print(f"{'='*80}\n")
                    # 保存最终模型
                    for i in range(N_AUV):
                        model_path = os.path.join(models_dir, f"DSAC_{i}_{ep}.pkl")
                        agents[i].save(model_path)
                    break
            
            # 每10个episode显示一次提前停止状态
            if ep % 10 == 0:
                print(f"  [Early Stop] Best: {best_ep_reward:.2f}, Patience: {patience_counter}/{early_stop_patience}")
        elif ep < early_stop_min_episodes and early_stop_enabled:
            # 记录历史，但还不启用提前停止
            if ep == early_stop_min_episodes - 1:
                window_size = min(10, len(ep_reward_history))
                best_ep_reward = np.mean(ep_reward_history[-window_size:])
                print(f"  [Early Stop] 已训练{early_stop_min_episodes}个episodes，开始监控性能提升...")
    
    # 训练结束
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"训练完成！")
    print(f"  总Episode数: {ep + 1}")
    print(f"  总步数: {total_steps}")
    print(f"  总时间: {elapsed_time/3600:.2f} 小时")
    print(f"{'='*80}")
    
    # 保存最终模型
    print(f"\n保存最终模型 (Episode {ep})...")
    for i in range(N_AUV):
        model_path = os.path.join(models_dir, f"DSAC_{i}_{ep}.pkl")
        agents[i].save(model_path)
        print(f"  ✓ 保存AUV{i}最终模型: {model_path}")
    
    print("\n训练完成！")
