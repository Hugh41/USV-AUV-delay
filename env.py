import numpy as np
import math
import copy
from tidewave_usbl import TideWave, USBL
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import time
from stackelberg_game import StackelbergGame


class Env(object):
    def __init__(self, args):
        # ---- paras args ----
        self.N_SNs = args.n_s
        self.N_AUV = args.N_AUV
        self.X_max = args.border_x
        self.Y_max = args.border_y
        self.border = np.array([self.X_max, self.Y_max])
        self.r_dc = args.R_dc
        self.N_POI = args.n_s
        self.epi_len = args.episode_length
        # ---- paras specified here ----
        self.USV_SHIFT_MAX = 4
        self.X_min = 0
        self.Y_min = 0
        self.r_dc = args.R_dc
        self.f = 20  # khz, AUV ~ SNs
        self.b = 1
        self.safe_dist = 10
        self.H = 100  # water depth
        self.V_max = 2.2
        self.V_min = 1.2
        self.S = 60
        self.P_u = 3e-2
        # ---- variables ----
        self.SoPcenter = np.zeros((self.N_POI, 2))  # center of SNs
        # 状态维度增加1，用于包含时间相位信息 (current_step % N) / N
        self.state_dim = 6 + 2 * (self.N_AUV - 1) + 1
        self.state = [np.zeros(self.state_dim)] * self.N_AUV
        self.rewards = []
        self.xy = np.zeros((self.N_AUV, 2))
        self.obs_xy = np.zeros((self.N_AUV, 2))  # observation
        self.usv_xy = np.zeros(2)
        self.vxy = np.zeros((self.N_AUV, 2))
        self.dis = np.zeros((self.N_AUV, self.N_POI))
        self.dis_hor = np.zeros((self.N_AUV, self.N_POI))  # horizontal distance
        # ---- SNs ----
        self.LDA = [3, 5, 8, 12]  # poisson variables
        CoLDA = np.random.randint(0, len(self.LDA), self.N_POI)
        self.lda = [self.LDA[CoLDA[i]] for i in range(self.N_POI)]  # assign poisson
        self.b_S = np.random.randint(0.0, 1000.0, self.N_POI).astype(np.float32)
        self.Fully_buffer = 5000
        self.H_Data_overflow = [0] * self.N_AUV
        self.Q = np.array(
            [self.lda[i] * self.b_S[i] / self.Fully_buffer for i in range(self.N_POI)]
        )
        self.idx_target = np.argsort(self.Q)[-self.N_AUV :]
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        # ---- Metrics ----
        self.FX = np.zeros(self.N_AUV)
        self.ec = np.zeros(self.N_AUV)
        self.TL = np.zeros(self.N_AUV)
        self.N_DO = 0
        self.crash = np.zeros(self.N_AUV)
        # ---- USBL ----
        self.usbl = USBL()
        # ---- TideWave ----
        self.tidewave = TideWave(self.H, self.X_max, self.Y_max, self.epi_len)
        self.tidewave.calc_tideWave()
        self.Ft = 0
        self.det_values = []  # 用于存储行列式值
        # ---- Stackelberg Game ----
        self.stackelberg_game = None  # 将在测试时设置agents后初始化
        self.use_stackelberg = False  # 是否使用Stackelberg博弈（默认关闭，训练时禁用，测试时显式启用）
        # ---- 时间尺度分离（Stackelberg博弈） ----
        # USV更新频率：USV每N个时间步更新一次（Leader慢）
        # 默认值：5（即USV每5个时间步更新一次，AUV每个时间步都更新）
        self.usv_update_frequency = getattr(args, 'usv_update_frequency', 5)
        self.last_usv_update_time = -1  # 上次USV更新的时间步
        # 存储AUV的历史状态和位置（用于USV决策时使用N步之前的状态）
        self.auv_history_states = []  # 历史状态列表
        self.auv_history_positions = []  # 历史位置列表
        self.max_history_length = self.usv_update_frequency + 1  # 保留足够的历史记录

    def calcnegdetJ_USV(self, posit_usv, auv_positions=None):
        """
        计算USV位置的负FIM行列式值
        
        Args:
            posit_usv: USV位置 [x, y]
            auv_positions: AUV位置数组，如果为None则使用self.xy（用于支持延迟状态）
        """
        if auv_positions is None:
            auv_positions = self.xy
        
        S_i = np.zeros(auv_positions.shape[0])
        p_i = np.zeros(auv_positions.shape[0])
        A_i = np.zeros(auv_positions.shape[0])
        # get the tidewave height
        pos_usv_3d = np.zeros(3)
        pos_usv_3d[:2] = posit_usv
        pos_usv_3d[2] = self.tidewave.get_tideHeight(
            posit_usv[0] / self.X_max, posit_usv[1] / self.Y_max, self.Ft
        )
        # we don't consider coeffs
        for i in range(auv_positions.shape[0]):
            pos_auv_3d = np.zeros(3)
            pos_auv_3d[:2] = auv_positions[i]
            S_i[i] = np.linalg.norm(pos_usv_3d - pos_auv_3d)
            p_i[i] = np.linalg.norm(pos_auv_3d)
            A_i[i] = (p_i[i] ** 4 - 2 * (S_i[i] ** 2) * (p_i[i] ** 2)) / (
                2 * (S_i[i] ** 6)
            )
        det_J1 = np.sum(S_i ** (-2))
        det_J2 = np.sum(2 * A_i + S_i ** (-2))
        det_J3 = 0
        for i in range(auv_positions.shape[0]):
            for j in range(i + 1, auv_positions.shape[0]):
                vi = auv_positions[i] - posit_usv
                vj = auv_positions[j] - posit_usv
                sinij = np.linalg.norm(np.cross(vi, vj)) / (
                    np.linalg.norm(vi) * np.linalg.norm(vj)
                )
                det_J3 += 4 * A_i[i] * A_i[j] * (sinij) ** 2

        # if any value is not reasonable, return 0
        if np.sum(np.isnan(np.array([det_J1, det_J2, det_J3]))) != 0:
            return 0
        else:
            det_value = -(det_J1 * det_J2 + det_J3)
            self.det_values.append(abs(det_value))  # 存储绝对值
            return det_value

    def set_agents(self, agents):
        """
        设置AUV智能体，用于Stackelberg博弈中的跟随者响应预测
        
        Args:
            agents: AUV的TD3智能体列表
        """
        self.stackelberg_game = StackelbergGame(self, agents)

    # bonus func: calculate optimal position for USV ()
    def calcposit_USV(self, use_delayed_state=True):
        """
        计算USV的最优位置
        如果启用了Stackelberg博弈，则考虑AUV的最优响应
        否则使用传统的优化方法
        
        Args:
            use_delayed_state: 是否使用延迟的状态（N步之前的AUV状态）
                              True: 使用历史状态（符合Stackelberg时间尺度分离）
                              False: 使用当前状态（仅用于初始化或特殊情况）
        """
        if self.use_stackelberg and self.stackelberg_game is not None:
            # 使用Stackelberg博弈求解
            init_guess = np.mean(self.xy, axis=0) if self.Ft == 0 else self.usv_xy
            
            # 时间尺度分离：根据use_delayed_state决定使用延迟状态还是当前状态
            if use_delayed_state and len(self.auv_history_states) > self.usv_update_frequency:
                # 使用N步之前的AUV状态（符合Stackelberg时间尺度分离：Leader慢，Follower快）
                # 注意：历史中已经包含了当前步的状态，所以要减1
                # 例如：当前是第t步，历史长度=t+1，要使用t-N步的状态
                # delayed_idx = (t+1) - 1 - N = t - N
                delayed_idx = len(self.auv_history_states) - 1 - self.usv_update_frequency
                # 确保索引有效
                delayed_idx = max(0, delayed_idx)
                delayed_auv_positions = self.auv_history_positions[delayed_idx]
                delayed_states = self.auv_history_states[delayed_idx]
                optimal_usv_pos, predicted_auv_pos, predicted_actions = self.stackelberg_game.solve_stackelberg(
                    current_auv_positions=delayed_auv_positions,
                    current_states=delayed_states,
                    init_guess=init_guess
                )
            else:
                # 使用当前AUV位置和状态（初始化或历史不足时）
                optimal_usv_pos, predicted_auv_pos, predicted_actions = self.stackelberg_game.solve_stackelberg(
                    current_auv_positions=self.xy,
                    current_states=self.state,
                    init_guess=init_guess
                )
            self.usv_xy = optimal_usv_pos
        else:
            # 使用传统方法（不考虑AUV响应，但使用延迟状态和更新频率）
            # 根据use_delayed_state决定使用延迟状态还是当前状态
            if use_delayed_state and len(self.auv_history_positions) > self.usv_update_frequency:
                # 使用N步之前的AUV位置（与传统方法延迟逻辑一致）
                delayed_idx = len(self.auv_history_positions) - 1 - self.usv_update_frequency
                delayed_idx = max(0, delayed_idx)
                delayed_auv_positions = self.auv_history_positions[delayed_idx]
                
                # 使用延迟位置计算USV位置
                init_guess = np.mean(delayed_auv_positions, axis=0) if self.Ft == 0 else self.usv_xy
                bounds = (
                    [
                        (init_guess[0] - self.X_max, init_guess[0] + self.X_max),
                        (init_guess[1] - self.Y_max, init_guess[1] + self.Y_max),
                    ]
                    if self.Ft == 0
                    else [
                        (
                            init_guess[0] - self.USV_SHIFT_MAX,
                            init_guess[0] + self.USV_SHIFT_MAX,
                        ),
                        (
                            init_guess[1] - self.USV_SHIFT_MAX,
                            init_guess[1] + self.USV_SHIFT_MAX,
                        ),
                    ]
                )
                tol = 1e-2 if self.Ft == 0 else 5e-2
                # 使用延迟位置计算FIM
                opt_asv_posit = differential_evolution(
                    lambda pos: self.calcnegdetJ_USV(pos, auv_positions=delayed_auv_positions),
                    bounds=bounds, tol=tol, maxiter=500
                )
                self.usv_xy = opt_asv_posit.x
            else:
                # 使用当前AUV位置（初始化或历史不足时）
                init_guess = np.mean(self.xy, axis=0) if self.Ft == 0 else self.usv_xy
                bounds = (
                    [
                        (init_guess[0] - self.X_max, init_guess[0] + self.X_max),
                        (init_guess[1] - self.Y_max, init_guess[1] + self.Y_max),
                    ]
                    if self.Ft == 0
                    else [
                        (
                            init_guess[0] - self.USV_SHIFT_MAX,
                            init_guess[0] + self.USV_SHIFT_MAX,
                        ),
                        (
                            init_guess[1] - self.USV_SHIFT_MAX,
                            init_guess[1] + self.USV_SHIFT_MAX,
                        ),
                    ]
                )
                tol = 1e-2 if self.Ft == 0 else 5e-2
                opt_asv_posit = differential_evolution(
                    self.calcnegdetJ_USV, bounds=bounds, tol=tol, maxiter=500
                )  # DE performs well in finding global solution
                self.usv_xy = opt_asv_posit.x

        # data rate calculating

    def calcRate(self, f, b, d, dir=0):
        f1 = (f - b / 2) if dir == 0 else (f + b / 2)
        lgNt = 17 - 30 * math.log10(f1)
        lgNs = 40 + 26 * math.log10(f1) - 60 * math.log10(f + 0.03)
        lgNw = 50 + 20 * math.log10(f1) - 40 * math.log10(f + 0.4)
        lgNth = -15 + 20 * math.log10(f1)
        NL = 10 * math.log10(
            1000
            * b
            * (
                10 ** (lgNt / 10)
                + 10 ** (lgNs / 10)
                + 10 ** (lgNw / 10)
                + 10 ** (lgNth / 10)
            )
        )
        alpha = (
            0.11 * ((f1**2) / (1 + f1**2))
            + 44 * ((f1**2) / (4100 + f1**2))
            + (2.75e-4) * (f1**2)
            + 0.003
        )
        TL = 15 * math.log10(d) + alpha * (0.001 * d)
        SL = 10 * math.log10(self.P_u) + 170.77
        R = 0.001 * b * math.log(1 + 10 ** (SL - TL - NL), 2)
        return R

    def get_state(self):  # new func
        for i in range(self.N_AUV):
            state = []
            # we assume that the AUVs cannot communicate directly
            # therefore, we measure the positions of AUVs by the AUV-USV communication
            meas_posit = np.zeros(3)
            meas_posit[:2] = self.xy[i]
            usv_xyz = np.zeros(3)
            usv_xyz[:2] = self.usv_xy
            usv_xyz[2] = self.tidewave.get_tideHeight(
                usv_xyz[0] / self.X_max, usv_xyz[1] / self.Y_max, self.Ft
            )
            usv_auv_diff = usv_xyz - meas_posit  # symmetry
            usv_auv_diff = self.usbl.calcPosit(usv_auv_diff, idx=i)
            meas_posit = usv_xyz - usv_auv_diff
            # input
            self.obs_xy[i][:2] = meas_posit[:2]
        # then get locs
        for i in range(self.N_AUV):
            state = []
            for j in range(self.N_AUV):
                if j == i:
                    continue
                state.append(
                    (self.obs_xy[j] - self.obs_xy[i]).flatten()
                    / np.linalg.norm(self.border)
                )
            # posit Target SNs
            state.append(
                (self.target_Pcenter[i] - self.obs_xy[i]).flatten()
                / np.linalg.norm(self.border)
            )
            state.append((self.obs_xy[i]).flatten() / np.linalg.norm(self.border))
            # finally, FX and N_DO
            state.append([self.FX[i] / self.epi_len, self.N_DO / self.N_POI])
            
            # 增加时间相位信息，解决POMDP问题
            # 归一化到 [0, 1]
            phase = (self.Ft % self.usv_update_frequency) / self.usv_update_frequency
            state.append([phase])
            
            self.state[i] = np.concatenate(tuple(state))

    # reset
    def reset(self):
        self.FX = np.zeros(self.N_AUV)
        self.ec = np.zeros(self.N_AUV)
        self.TL = np.zeros(self.N_AUV)
        self.N_DO = 0
        self.crash = np.zeros(self.N_AUV)
        # assign x/y to SNs
        self.SoPcenter[:, 0] = np.random.randint(
            self.safe_dist, self.X_max - self.safe_dist, size=self.N_POI
        )
        self.SoPcenter[:, 1] = np.random.randint(
            self.safe_dist, self.Y_max - self.safe_dist, size=self.N_POI
        )
        # assign x/y to AUVs, the distance between AUVs > 2 * safe_dist
        while True:
            dist_ok = True
            self.xy[:,0] = np.random.randint(
                self.safe_dist, self.X_max - self.safe_dist, size=self.N_AUV
            )
            self.xy[:,1] = np.random.randint(
                self.safe_dist, self.Y_max - self.safe_dist, size=self.N_AUV
            )
            for i in range(self.N_AUV):
                for j in range(i + 1, self.N_AUV):
                    if np.linalg.norm(self.xy[i] - self.xy[j]) < 2 * self.safe_dist:
                        dist_ok = False
            if dist_ok == True:
                break
        # reset the position of ASV（初始化时使用当前状态）
        self.calcposit_USV(use_delayed_state=False)
        self.b_S = np.random.randint(0, 1000, self.N_POI)
        # assign target SNs
        self.Q = np.array(
            [self.lda[i] * self.b_S[i] / self.Fully_buffer for i in range(self.N_POI)]
        )
        self.idx_target = np.argsort(self.Q)[-self.N_AUV :]
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.target_Pcenter = self.SoPcenter[self.idx_target]

        # states
        self.get_state()
        # 初始化历史记录（传统方法和Stackelberg方法都需要，用于延迟状态）
        self.last_usv_update_time = -1
        self.auv_history_states = []
        self.auv_history_positions = []
        # 记录初始状态（传统方法和Stackelberg方法都需要）
        self.auv_history_states.append(copy.deepcopy(self.state))
        self.auv_history_positions.append(copy.deepcopy(self.xy))
        return self.state

    def posit_change(self, actions, hovers):
        # 记录历史（传统方法和Stackelberg方法都需要，用于延迟状态）
        # 必须在位置更新前记录以保持状态-位置一致性
        self.auv_history_states.append(copy.deepcopy(self.state))
        self.auv_history_positions.append(copy.deepcopy(self.xy))
        # 只保留最近的历史记录
        if len(self.auv_history_states) > self.max_history_length:
            self.auv_history_states.pop(0)
            self.auv_history_positions.pop(0)

        for i in range(self.N_AUV):
            # action mapping
            actions[i][0] = 0.5 * (actions[i][0] + 1)
            detX = (actions[i][0] * (self.V_max - self.V_min) + self.V_min) * math.cos(
                actions[i][1] * math.pi
            )
            detY = (actions[i][0] * (self.V_max - self.V_min) + self.V_min) * math.sin(
                actions[i][1] * math.pi
            )
            self.vxy[i, 0] = detX
            self.vxy[i, 1] = detY
            V = math.sqrt(pow(detX, 2) + pow(detY, 2))
            if hovers[i] == True:
                detX = 0
                detY = 0
            xy_ = copy.deepcopy(self.xy[i])
            xy_[0] += detX
            xy_[1] += detY
            # getting the metric of crossing the border
            Flag = False
            self.FX[i] = (
                np.sum((xy_ - np.array([0, 0])) < 0) + np.sum((self.border - xy_) < 0)
            ) > 0
            Flag = (np.sum((xy_) < 0) + np.sum((self.border - xy_) < 0)) == 0
            if not Flag:  # Flag False -> cross the border
                xy_[0] -= detX
                xy_[1] -= detY
            if Flag and (hovers[i] == False):
                F = (0.7 * self.S * (V**2)) / 2
                self.ec[i] = (F * V) / (
                    -0.081 * (V**3) + 0.215 * (V**2) - 0.01 * V + 0.541
                ) + 15
            else:
                self.ec[i] = 90 + 15
            # assigning positions
            self.xy[i] = xy_
        
        # 关键修正：在更新USV之前，先更新一次观测状态
        # 这样USV在做决策时，能看到AUV的新位置（虽然USV位置还是旧的）
        # 这解决了"位置已更新但状态仍是旧的"的问题
        self.get_state()

        # 时间尺度分离：USV每N个时间步更新一次（Leader慢，Follower快）
        if self.use_stackelberg and self.stackelberg_game is not None:
            # Stackelberg方法：USV每usv_update_frequency个时间步更新一次
            should_update_usv = (self.Ft % self.usv_update_frequency == 0)
            if should_update_usv:
                # 使用延迟状态（N步之前的AUV状态）
                self.calcposit_USV(use_delayed_state=True)
                self.last_usv_update_time = self.Ft
            # 如果不需要更新，USV位置保持不变（使用上次更新的位置）
        else:
            # 传统方法：实时更新（每步都更新，使用当前状态）
            # 与Stackelberg方法不同，传统方法使用实时状态进行优化
            self.calcposit_USV(use_delayed_state=False)


    def step_move(self, hovers):
        self.N_DO = 0
        self.b_S += [np.random.poisson(self.lda[i]) for i in range(self.N_POI)]
        for i in range(self.N_POI):  # check data overflow
            if self.b_S[i] >= self.Fully_buffer:
                self.N_DO += 1
                self.b_S[i] = self.Fully_buffer
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.crash = np.zeros(self.N_AUV)
        self.TL = np.zeros(self.N_AUV)
        self.rewards = np.zeros(self.N_AUV)
        data_rate = np.zeros(self.N_AUV)
        # get state
        self.get_state()
        # get crash information
        for i in range(self.N_AUV):
            for j in range(self.N_AUV):
                if j == i:
                    continue
                dxy = (self.xy[j] - self.xy[i]).flatten()
                sd = np.linalg.norm(dxy)
                if sd < 5:
                    self.crash[i] += 1
            # then calculating dis AUV ~ target SNs
            self.calc_dist(i)
            if self.dis_hor[i, self.idx_target[i]] < self.r_dc:
                self.TL[i] = True
                data_rate[i] = max(
                    self.calcRate(self.f, self.b, self.dis[i, self.idx_target[i]], 0),
                    self.calcRate(self.f, self.b, self.dis[i, self.idx_target[i]], 1),
                )
                self.b_S[self.idx_target[i]] = 0
            self.rewards = self.compute_reward()
        return self.state, self.rewards, self.TL, data_rate, self.ec, self.crash

    def calc_dist(self, idx):
        # get height
        H = self.tidewave.get_tideHeight(
            self.xy[idx][0] / self.X_max, self.xy[idx][1] / self.Y_max, self.Ft
        )
        for i in range(self.N_POI):
            self.dis[idx][i] = math.sqrt(
                pow(self.SoPcenter[i][0] - self.xy[idx][0], 2)
                + pow(self.SoPcenter[i][1] - self.xy[idx][1], 2)
                + pow(self.H, 2)
            )
            self.dis_hor[idx][i] = math.sqrt(
                pow(self.SoPcenter[i][0] - self.xy[idx][0], 2)
                + pow(self.SoPcenter[i][1] - self.xy[idx][1], 2)
            )

    def CHOOSE_AIM(self, idx=0, lamda=0.05):
        self.calc_dist(idx=idx)
        Q = np.array(
            [
                self.lda[i] * self.b_S[i] / self.Fully_buffer - lamda * self.dis[idx][i]
                for i in range(self.N_POI)
            ]
        )
        idx_target = np.argsort(Q)[-self.N_AUV :]
        inter = np.intersect1d(idx_target, self.idx_target)
        if len(inter) < len(self.idx_target):
            diff = np.setdiff1d(idx_target, inter)
            self.idx_target[idx] = diff[0]
        else:
            idx_target = np.argsort(self.Q)[-(self.N_AUV + 1) :]
            self.idx_target[idx] = idx_target[0]
        self.target_Pcenter = self.SoPcenter[self.idx_target]
        # state[i]
        st_idx = 2 * (self.N_AUV - 1)
        self.state[idx][st_idx : st_idx + 2] = (
            self.target_Pcenter[idx] - self.xy[idx]
        ).flatten() / np.linalg.norm(self.border)
        self.state[idx][-2] = self.N_DO / self.N_POI # 倒数第二个是N_DO
        self.state[idx][-1] = (self.Ft % self.usv_update_frequency) / self.usv_update_frequency # 最后一个是phase
        return self.state[idx]

    def compute_reward(self):  # oracle
        reward = np.zeros(self.N_AUV)
        for i in range(self.N_AUV):
            dist_to_target = np.linalg.norm(self.xy[i] - self.target_Pcenter[i])
            reward[i] += -0.6 * dist_to_target - self.FX[i] * 0.1 - self.N_DO * 0.05
            for j in range(i + 1, self.N_AUV):
                dist_between_auvs = np.linalg.norm(self.xy[j] - self.xy[i])
                if dist_between_auvs < 12:
                    reward[i] -= 6 * (12 - dist_between_auvs)
            # rew
            if self.TL[i] > 0:
                reward[i] += 12
            reward[i] -= 0.085 * self.ec[i]  # adjust this factor
        return reward
