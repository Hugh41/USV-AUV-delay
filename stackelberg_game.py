"""
Stackelberg博弈求解器
领导者（USV）与跟随者（AUV）的博弈框架

核心原理：
- 领导者（USV）先选择策略（位置）
- 跟随者（AUV）在预见到领导者最优回应的情况下，选择对自己最有利的策略
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import copy


class StackelbergGame:
    """
    Stackelberg博弈求解器类
    """
    
    def __init__(self, env, agents=None, lambda_J=1.0, lambda_u=0.05, fast_mode=False):
        """
        Args:
            env: environment object
            agents: list of AUV TD3 agents for follower response prediction
            lambda_J: FIM geometry quality weight
            lambda_u: USV motion regularization weight (paper Eq. 24)
            fast_mode: if True, use reduced DE budget (for training speed)
        """
        self.env = env
        self.agents = agents
        self.lambda_J = lambda_J
        self.lambda_u = lambda_u
        self.fast_mode = fast_mode
        self._prev_usv_xy = None
        
    def follower_best_response(self, usv_position, current_auv_positions, current_states):
        """
        计算跟随者（AUV）在给定USV位置下的最优响应
        
        Args:
            usv_position: USV的位置 [x, y]
            current_auv_positions: 当前AUV位置列表
            current_states: 当前AUV状态列表
            
        Returns:
            predicted_auv_positions: 预测的AUV最优位置
            predicted_actions: 预测的AUV最优动作
        """
        if self.agents is None:
            # 如果没有智能体，使用简单的启发式方法
            # 假设AUV会朝着目标点移动
            predicted_positions = copy.deepcopy(current_auv_positions)
            for i in range(len(current_auv_positions)):
                # 简单的预测：AUV会朝着目标点移动一定距离
                target = self.env.target_Pcenter[i] if hasattr(self.env, 'target_Pcenter') else current_auv_positions[i]
                direction = target - current_auv_positions[i]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                # 预测下一步位置（假设以中等速度移动）
                v_avg = (self.env.V_max + self.env.V_min) / 2
                predicted_positions[i] = current_auv_positions[i] + direction * v_avg
            return predicted_positions, None
        else:
            # 使用TD3智能体预测AUV的最优动作
            predicted_positions = []
            predicted_actions = []
            
            # 临时更新USV位置以计算状态
            original_usv_xy = copy.deepcopy(self.env.usv_xy)
            self.env.usv_xy = usv_position
            
            # 重新计算状态（基于新的USV位置）
            temp_states = []
            for i in range(len(current_auv_positions)):
                # 计算基于新USV位置的状态
                state = self._compute_state_for_auv(i, current_auv_positions, usv_position)
                temp_states.append(state)
                
                # 使用智能体选择动作
                if i < len(self.agents):
                    action = self.agents[i].select_action(state)
                    predicted_actions.append(action)
                    
                    # 将动作转换为位置变化
                    action_mapped = copy.deepcopy(action)
                    action_mapped[0] = 0.5 * (action_mapped[0] + 1)
                    detX = (action_mapped[0] * (self.env.V_max - self.env.V_min) + self.env.V_min) * np.cos(
                        action_mapped[1] * np.pi
                    )
                    detY = (action_mapped[0] * (self.env.V_max - self.env.V_min) + self.env.V_min) * np.sin(
                        action_mapped[1] * np.pi
                    )
                    
                    # 预测新位置
                    new_pos = current_auv_positions[i] + np.array([detX, detY])
                    # 边界检查
                    new_pos = np.clip(new_pos, [0, 0], self.env.border)
                    predicted_positions.append(new_pos)
                else:
                    predicted_positions.append(current_auv_positions[i])
                    predicted_actions.append(np.array([0, 0]))
            
            # 恢复原始USV位置
            self.env.usv_xy = original_usv_xy
            
            return np.array(predicted_positions), predicted_actions
    
    def _compute_state_for_auv(self, idx, auv_positions, usv_position):
        """
        为特定AUV计算状态（基于给定的USV位置）
        模拟env.get_state()的逻辑，但简化USBL测量部分
        
        Args:
            idx: AUV索引
            auv_positions: AUV位置列表
            usv_position: USV位置
            
        Returns:
            state: AUV的状态向量
        """
        state = []
        
        # 模拟USBL测量（简化版本：直接使用位置，添加一些噪声模拟测量误差）
        # 在实际环境中，USBL会通过USV-AUV通信测量位置
        meas_positions = copy.deepcopy(auv_positions)
        
        # 其他AUV的相对位置（基于测量位置）
        # 注意：必须使用.flatten()确保是1维数组，与env.get_state()保持一致
        for j in range(len(meas_positions)):
            if j == idx:
                continue
            rel_pos = (meas_positions[j] - meas_positions[idx]) / np.linalg.norm(self.env.border)
            state.append(rel_pos.flatten())  # 确保展平为1维
        
        # 目标点的相对位置
        if hasattr(self.env, 'target_Pcenter') and idx < len(self.env.target_Pcenter):
            target_rel = (self.env.target_Pcenter[idx] - meas_positions[idx]) / np.linalg.norm(self.env.border)
            state.append(target_rel.flatten())  # 确保展平为1维
        else:
            state.append(np.array([0, 0]).flatten())  # 确保展平为1维
        
        # 自身位置（归一化）
        self_pos = meas_positions[idx] / np.linalg.norm(self.env.border)
        state.append(self_pos.flatten())  # 确保展平为1维
        
        # FX和N_DO（使用当前值）
        fx = self.env.FX[idx] / self.env.epi_len if hasattr(self.env, 'FX') and idx < len(self.env.FX) else 0
        n_do = self.env.N_DO / self.env.N_POI if hasattr(self.env, 'N_DO') else 0
        state.append(np.array([fx, n_do]))
        
        # 增加时间相位信息，解决维度不匹配问题
        # 归一化到 [0, 1]
        phase = (self.env.Ft % self.env.usv_update_frequency) / self.env.usv_update_frequency
        state.append(np.array([phase]))
        
        return np.concatenate(state)
    
    def leader_objective(self, usv_position, current_auv_positions, current_states):
        """
        领导者（USV）的目标函数（含运动正则化项）

        最大化: λ_J * det(J(p_u, P_br)) - λ_u * ||p_u - p_{u,t-1}||²  (论文 Eq. 24)
        等价于最小化其负值，即返回:
            λ_J * neg_detJ + λ_u * ||p_u - prev_usv||²

        Args:
            usv_position: USV的候选位置 [x, y]
            current_auv_positions: 当前AUV位置
            current_states: 当前AUV状态
            
        Returns:
            scalar: 最小化目标值（越小越好）
        """
        # 计算跟随者的最优响应
        predicted_auv_positions, _ = self.follower_best_response(
            usv_position, current_auv_positions, current_states
        )
        
        # 临时保存原始AUV位置
        original_auv_positions = copy.deepcopy(self.env.xy)
        
        # 临时更新AUV位置为预测位置，以便计算detJ
        self.env.xy = predicted_auv_positions
        
        # 基于预测的AUV位置计算Fisher信息矩阵的detJ
        neg_detJ = self.env.calcnegdetJ_USV(usv_position)
        
        # 恢复原始AUV位置
        self.env.xy = original_auv_positions

        # 运动正则化项：λ_u * ||p_u - p_{u,t-1}||²  (对应论文 Eq. 24)
        if self._prev_usv_xy is not None:
            motion_penalty = self.lambda_u * np.linalg.norm(
                np.array(usv_position) - np.array(self._prev_usv_xy)
            ) ** 2
        else:
            motion_penalty = 0.0

        return self.lambda_J * neg_detJ + motion_penalty
    
    def solve_stackelberg(self, current_auv_positions, current_states, init_guess=None):
        """
        求解Stackelberg博弈均衡
        
        Args:
            current_auv_positions: 当前AUV位置列表
            current_states: 当前AUV状态列表
            init_guess: 初始猜测的USV位置（可选）
            
        Returns:
            optimal_usv_position: 最优USV位置
            predicted_auv_positions: 预测的AUV最优位置
            predicted_actions: 预测的AUV最优动作
        """
        # 确定搜索边界
        if init_guess is None:
            init_guess = np.mean(current_auv_positions, axis=0) if len(current_auv_positions) > 0 else self.env.usv_xy
        
        if self.env.Ft == 0:
            # 初始化阶段：全局搜索，不施加运动正则化
            self._prev_usv_xy = None
            bounds = [
                (0, self.env.X_max),
                (0, self.env.Y_max)
            ]
        else:
            # 动态调整搜索范围：如果之前没动，这次尝试多看一点点，防止死锁
            search_range = self.env.USV_SHIFT_MAX
            bounds = [
                (init_guess[0] - search_range, init_guess[0] + search_range),
                (init_guess[1] - search_range, init_guess[1] + search_range)
            ]
        
        # 定义目标函数（考虑跟随者响应）
        def objective(usv_pos):
            return self.leader_objective(usv_pos, current_auv_positions, current_states)
        
        # fast_mode: lightweight budget for training; full budget for evaluation
        if self.fast_mode:
            de_kwargs = dict(tol=0.05, popsize=5, maxiter=20, seed=None)
        else:
            de_kwargs = dict(tol=0.01, popsize=20, mutation=(0.5, 1.0),
                             recombination=0.7, maxiter=100, seed=None)

        result = differential_evolution(objective, bounds=bounds, **de_kwargs)
        
        optimal_usv_position = result.x

        # 更新 prev_usv_xy 以供下次调用的运动正则化项使用
        self._prev_usv_xy = copy.deepcopy(optimal_usv_position)
        
        # 计算在最优USV位置下，AUV的最优响应
        predicted_auv_positions, predicted_actions = self.follower_best_response(
            optimal_usv_position, current_auv_positions, current_states
        )
        
        return optimal_usv_position, predicted_auv_positions, predicted_actions

