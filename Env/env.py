import numpy as np
import cvxopt as cvx
import pandas as pd
import math
from itertools import product

from Env.opter import CvxOpt

class Env:

    REWARD_NEG = 0
    STATE_3 = 3
    STATE_2 = 2
    STATE_ON = 1
    STATE_OFF = 0

    def __init__(self, name, configure):

        self.name = name

        if configure.random_seed >= 0:
            np.random.seed(configure.random_seed)

        self._num_core = 4
        self._num_rrh = configure.num_rrh
        self._num_usr = configure.num_usr

        self._DM_MAX = configure.demand_max
        self._DM_MIN = configure.demand_min

        self._pow_on = configure.pow_on
        self._pow_slp = configure.pow_slp
        self._pow_gap = configure.pow_gap
        self._pow_tsm = configure.pow_tsm

        # Start JaeHyun Code
        self.cnt_task = self._num_rrh
        self.index_optimal_core = self._num_core

        self._task_util = [0.05, 0.11, 0.05, 0.43, 0.17, 0.11, 0.08, 0.02, 0.12, 0.18, 0.07, 0.09]
        #self._task_util = [0.1]*12
        self._core0_load = 0
        self._core1_load = 0
        self._core2_load = 0
        self._core3_load = 0
        self._core_load = [0]*self._num_core

        self._reward_util = configure.reward_util
        self._load_diff = configure.load_diff
        self._task_num = configure.task_num
        self._load_std = 0

        #self._task_util = np.random.uniform(self._DM_MIN, self._DM_MAX, size=self._num_rrh)
        print("*************mean of this", np.mean(self._task_util))

        self.E2E_Deadline = 8
        # Num          = {1,    2,    3,    4,    5,    6,    7,    8,   9,    10,   11,   12  }
        self.TASK_Exec = [0.55, 0.15, 0.22, 0.23, 0.19, 0.78, 0.32, 2.6, 0.48, 0.18, 0.32, 0.21]
        # self._task_util = [0.05, 0.11, 0.05, 0.11, 0.17, 0.35, 0.08, 0.02, 0.12, 0.08, 0.07, 0.09]
        # self.TASK_Exec = [0.55, 0.15, 0.22, 0.23, 0.19, 0.78, 0.32, 2.6, 0.48, 0.18, 0.32, 0.21]

        self.Task_Load = [0] * self._num_rrh
        self.Task_Window = [0] * 12
        self.Task_Offset = [0] * 12
        self.Task_Deadline = [0] * 12

        self.Chain1_Offset = [0] * 12
        self.Chain1_Deadline = [100] * 12
        self.Chain2_Offset = [0] * 12
        self.Chain2_Deadline = [100] * 12
        self.Chain3_Offset = [0] * 12
        self.Chain3_Deadline = [100] * 12
        self.Chain4_Offset = [0] * 12
        self.Chain4_Deadline = [100] * 12

        self.chain1_buffer = [0] * 12
        self.chain2_buffer = [0] * 12
        self.chain3_buffer = [0] * 12
        self.chain4_buffer = [0] * 12

        self.ScheduleError = 0
        self.ScheduleOkay = 0

        self._load_max = 1
        self._load_min = 0

        # End JaeHyun Code

        self._dm = self._generate_demand()

        self.MAX_EP = configure.episodes
        self.MAX_EXP_EP = configure.epsilon_steps
        self.MAX_TEST_EP = configure.tests

        self._dm_index = 0

        self._BAND = configure.band   #######bandwidth
        self._ETA = configure.eta     
        self._THETA_2 = configure.theta_2
        self._TM = configure.tm
        # todo replace const with dynamical variable
        self._CONST = 1.345522816371604e-06

        self._P_MIN, self._P_MAX = self._get_power_bound()

        all_off = np.zeros(self._num_rrh) + self.STATE_OFF
        self._state_rrh_min = all_off.copy()
        self._state_rrh_min_last = all_off.copy()
        self._state_rrh_max = all_off.copy()
        self._state_rrh_last = self._state_rrh = all_off.copy()
        self._state_rrh_rd_last = self._state_rrh_rd = all_off.copy()

        self.reset()
        self.find_optimal_value()

    @property           ### state space is the user demand plus the number of rrh
    def state(self):
        # dm = (self._demand - self._DM_MIN) / (self._DM_MAX - self._DM_MIN)
        dm = abs(self._core1_load - self._core0_load)

        print("state", self._state_rrh)
        # print("dm", dm)
        # print("Total:", self._demand)

        return self._state_rrh
        # return np.concatenate([self._state_rrh, dm])                    ####Concatenation refers to joining. This function is used to join two or more arrays of the same shape along a specified axis

    @property
    def demand(self):
        return np.around(self._demand / 10e6, decimals=3)

    @property
    def dim_state(self):
        return len(self.state)

    @property
    def dim_action(self):
        return self._num_rrh * self._num_core + 1
        # return self._num_rrh + 1

    @property
    def num_rrh(self):
        return self._num_rrh

    @property
    def num_rrh_on(self):
        return len((np.where(self._state_rrh == self.STATE_ON))[0])

    @property
    def max_rrh_reward(self):
        return self.on_max, self.power_max, self.reward_max

    @property
    def min_rrh_reward(self):
        return self.on_min, self.power_min, self.reward_min

    @property
    def rnd_rrh_reward(self):
        self._core2_load = 0
        self._core1_load = 0
        self._core0_load = 0

        return self.on_rnd, self.power_rnd, self.reward_rnd

    def run_fix_solution(self):
        self._get_max_rrh_solution()
        self._get_min_rrh_solution()
        self._get_rnd_rrh_solution()
        self.on_max, self.power_max, self.reward_max = self._get_max_rrh_reward()
        self.on_min, self.power_min, self.reward_min = self._get_min_rrh_reward()
        self.on_rnd, self.power_rnd, self.reward_rnd = self._get_rnd_rrh_reward()

        self.util_diff()
        #self.set_task_graph()

        self._reward_util = self._get_max_util_reward()
        print("Reward Value:", self._reward_util)

    def reward_to_power(self, reward):
        return (1.0 - reward) * (self._P_MAX - self._P_MIN) + self._P_MIN

    def reset(self):
        self.reset_channel()
        self.reset_demand()
        self.run_fix_solution()
        self.util_diff()
        s = self.reset_state()  # 이거 고쳐야 함 ****
        return s

    def find_optimal_value(self):
        buffer = [0]*100000000
        cnt = 0
        lst_core = list(range(1, self._num_core+1))

        print("모든 조합의 경우의수:")
        for i in product(lst_core, repeat=self._num_rrh):
            self.optimal_util_diff(i)
            buffer[cnt] = self._load_std
            cnt = cnt + 1
            print("*****Count: *********", cnt)

        buffer = min(buffer[:cnt-1])
        print("*****Optimize Value is *********", buffer)

    def util_diff(self):
        if self._num_core == 4:
            for i in range(len(self._state_rrh)):
                if self._state_rrh[i] == 0:
                    self._core0_load += self._task_util[i]
                elif self._state_rrh[i] == 1:
                    self._core1_load += self._task_util[i]
                elif self._state_rrh[i] == 2:
                    self._core2_load += self._task_util[i]
                elif self._state_rrh[i] == 3:
                    self._core3_load += self._task_util[i]

            log_std = [self._core0_load, self._core1_load, self._core2_load, self._core3_load]

        elif self._num_core == 3:
            for i in range(len(self._state_rrh)):
                if self._state_rrh[i] == 0:
                    self._core0_load += self._task_util[i]
                elif self._state_rrh[i] == 1:
                    self._core1_load += self._task_util[i]
                elif self._state_rrh[i] == 2:
                    self._core2_load += self._task_util[i]

            log_std = [self._core0_load, self._core1_load, self._core2_load]

        elif self._num_core == 2:
            for i in range(len(self._state_rrh)):
                if self._state_rrh[i] == 0:
                    self._core0_load += self._task_util[i]
                elif self._state_rrh[i] == 1:
                    self._core1_load += self._task_util[i]

            log_std = [self._core0_load, self._core1_load]

        self._load_std = np.std(log_std)

        print("State_check:", self._state_rrh)
        print("Core2_Load:", self._core2_load)
        print("Core1_Load:", self._core1_load)
        print("Core0_Load:", self._core0_load)
        print("Load_Variance:", self._load_std)

    def optimal_util_diff(self, task):
        _log_std = 0

        if self._num_core == 4:
            for i in range(len(task)):
                if task[i] == 0:
                    self._core0_load += self._task_util[i]
                elif task[i] == 1:
                    self._core1_load += self._task_util[i]
                elif task[i] == 2:
                    self._core2_load += self._task_util[i]
                elif task[i] == 3:
                    self._core3_load += self._task_util[i]

            _log_std = [self._core0_load, self._core1_load, self._core2_load, self._core3_load]

        elif self._num_core == 3:
            for i in range(len(task)):
                if task[i] == 0:
                    self._core0_load += self._task_util[i]
                elif task[i] == 1:
                    self._core1_load += self._task_util[i]
                elif task[i] == 2:
                    self._core2_load += self._task_util[i]

            _log_std = [self._core0_load, self._core1_load, self._core2_load]

        elif self._num_core == 2:
            for i in range(len(task)):
                if task[i] == 0:
                    self._core0_load += self._task_util[i]
                elif task[i] == 1:
                    self._core1_load += self._task_util[i]

            _log_std = [self._core0_load, self._core1_load]

        return np.std(_log_std)

    def set_task_graph(self):

        # BSW_100ms = 0
        # COM_RTE = 1
        # ASW_1000ms = 2
        # ASW_10ms_2 = 3
        # ASW_100ms_2 = 4
        # ASW_100ms = 5
        # ASW_20ms = 6
        # ASW_10ms = 7
        # ASW_50ms = 8
        # ASW_50ms_2 = 9
        # BSW_10ms = 10
        # BSW_5ms = 11
        # Num            = {0,    1,    2,    3,    4,    5,    6,    7,   8,    9,    10,   11  }
        # self.TASK_Exec = {0.55, 0.15, 0.22, 0.23, 0.19, 0.78, 0.32, 2.6, 0.48, 0.18, 0.32, 0.21}
        # self.Task_Load = [0] * 12

        # Task-Mapping
        for i in range(len(self._state_rrh)):
            if self._state_rrh[i] == 1:
                self.Task_Load[i] = self._core1_load
            else:
                self.Task_Load[i] = self._core0_load

        # print(self.TASK_Exec)
        # print(self.Task_Load)

        Chain1_k = self.E2E_Deadline/(self.TASK_Exec[0] * self.Task_Load[0] + self.TASK_Exec[1] * self.Task_Load[1] +
                                      self.TASK_Exec[2] * self.Task_Load[2] + self.TASK_Exec[3] * self.Task_Load[3] +
                                      self.TASK_Exec[4] * self.Task_Load[4] + self.TASK_Exec[5] * self.Task_Load[5])

        Chain2_k = self.E2E_Deadline / (self.TASK_Exec[0] * self.Task_Load[0] + self.TASK_Exec[1] * self.Task_Load[1] +
                                        self.TASK_Exec[2] * self.Task_Load[2] + self.TASK_Exec[3] * self.Task_Load[3] +
                                        self.TASK_Exec[9] * self.Task_Load[9] + self.TASK_Exec[5] * self.Task_Load[5])

        Chain3_k = self.E2E_Deadline / (self.TASK_Exec[0] * self.Task_Load[0] + self.TASK_Exec[1] * self.Task_Load[1] +
                                        self.TASK_Exec[6] * self.Task_Load[6] + self.TASK_Exec[7] * self.Task_Load[7] +
                                        self.TASK_Exec[8] * self.Task_Load[8] + self.TASK_Exec[3] * self.Task_Load[3] +
                                        self.TASK_Exec[9] * self.Task_Load[9] + self.TASK_Exec[5] * self.Task_Load[5])

        Chain4_k = self.E2E_Deadline / (self.TASK_Exec[0] * self.Task_Load[0] + self.TASK_Exec[1] * self.Task_Load[1] +
                                        self.TASK_Exec[6] * self.Task_Load[6] + self.TASK_Exec[7] * self.Task_Load[7] +
                                        self.TASK_Exec[8] * self.Task_Load[8] + self.TASK_Exec[3] * self.Task_Load[3] +
                                        self.TASK_Exec[4] * self.Task_Load[4] + self.TASK_Exec[5] * self.Task_Load[5])

        for i in range(0, 6):
            self.Task_Window[i] = self.TASK_Exec[i] * self.Task_Load[i] * Chain1_k
            self.chain1_buffer[i] = self.Task_Window[i]
            if i > 0:
                for j in range(0, i):
                    self.Chain1_Offset[i] += self.chain1_buffer[j]
            self.Chain1_Deadline[i] = self.Chain1_Offset[i] + self.chain1_buffer[i]

        # print("Chain1_Check Offset:", self.chain1_buffer)
        # print("Chain1_Check Offset:", self.Chain1_Offset)
        # print("Chain1_Check Offset:", self.Chain1_Deadline)

        for i in range(0, 4):
            self.Task_Window[i] = self.TASK_Exec[i] * self.Task_Load[i] * Chain2_k
            self.chain2_buffer[i] = self.Task_Window[i]
            if i > 0:
                for j in range(0, i):
                    self.Chain2_Offset[i] += self.chain2_buffer[j]
            self.Chain2_Deadline[i] = self.Chain2_Offset[i] + self.chain2_buffer[i]

        self.chain2_buffer[9] = self.TASK_Exec[9] * self.Task_Load[9] * Chain2_k
        self.Chain2_Offset[9] = self.Chain2_Deadline[3]
        self.Chain2_Deadline[9] = self.Chain2_Offset[9] + self.chain2_buffer[9]

        self.chain2_buffer[5] = self.TASK_Exec[5] * self.Task_Load[5] * Chain2_k
        self.Chain2_Offset[5] = self.Chain2_Deadline[9]
        self.Chain2_Deadline[5] = self.Chain2_Offset[5] + self.chain2_buffer[5]

        # print("Chain2_Check Offset:", self.chain2_buffer)
        # print("Chain2_Check Offset:", self.Chain2_Offset)
        # print("Chain2_Check Offset:", self.Chain2_Deadline)

        # 0 -> 1 -> 6 -> 7 -> 8 -> 3 -> 9 -> 5
        for i in range(0, 5):
            self.chain3_buffer[i+5] = self.TASK_Exec[i+5] * self.Task_Load[i+5] * Chain3_k
        self.chain3_buffer[0] = self.TASK_Exec[0] * self.Task_Load[0] * Chain3_k
        self.chain3_buffer[1] = self.TASK_Exec[1] * self.Task_Load[1] * Chain3_k
        self.chain3_buffer[3] = self.TASK_Exec[3] * self.Task_Load[3] * Chain3_k

        self.Chain3_Deadline[0] = self.Chain3_Offset[0] + self.chain3_buffer[0]
        self.Chain3_Offset[1] = self.Chain3_Deadline[0]
        self.Chain3_Deadline[1] = self.Chain3_Offset[1] + self.chain3_buffer[1]
        self.Chain3_Offset[6] = self.Chain3_Deadline[1]
        self.Chain3_Deadline[6] = self.Chain3_Offset[6] + self.chain3_buffer[6]
        self.Chain3_Offset[7] = self.Chain3_Deadline[6]
        self.Chain3_Deadline[7] = self.Chain3_Offset[7] + self.chain3_buffer[7]
        self.Chain3_Offset[8] = self.Chain3_Deadline[7]
        self.Chain3_Deadline[8] = self.Chain3_Offset[8] + self.chain3_buffer[8]
        self.Chain3_Offset[3] = self.Chain3_Deadline[8]
        self.Chain3_Deadline[3] = self.Chain3_Offset[3] + self.chain3_buffer[3]
        self.Chain3_Offset[9] = self.Chain3_Deadline[3]
        self.Chain3_Deadline[9] = self.Chain3_Offset[9] + self.chain3_buffer[9]
        self.Chain3_Offset[5] = self.Chain3_Deadline[9]
        self.Chain3_Deadline[5] = self.Chain3_Offset[5] + self.chain3_buffer[5]

        # print("Chain3_Check Offset:", self.chain3_buffer)
        # print("Chain3_Check Offset:", self.Chain3_Offset)
        # print("Chain3_Check Offset:", self.Chain3_Deadline)

        # 0 -> 1 -> 6 -> 7 -> 8 -> 3 -> 4 -> 5
        for i in range(0, 5):
            self.chain4_buffer[i+4] = self.TASK_Exec[i+4] * self.Task_Load[i+4] * Chain4_k
        self.chain4_buffer[0] = self.TASK_Exec[0] * self.Task_Load[0] * Chain4_k
        self.chain4_buffer[1] = self.TASK_Exec[1] * self.Task_Load[1] * Chain4_k
        self.chain4_buffer[3] = self.TASK_Exec[3] * self.Task_Load[3] * Chain4_k

        self.Chain4_Deadline[0] = self.Chain4_Offset[0] + self.chain4_buffer[0]
        self.Chain4_Offset[1] = self.Chain4_Deadline[0]
        self.Chain4_Deadline[1] = self.Chain4_Offset[1] + self.chain4_buffer[1]
        self.Chain4_Offset[6] = self.Chain4_Deadline[1]
        self.Chain4_Deadline[6] = self.Chain4_Offset[6] + self.chain4_buffer[6]
        self.Chain4_Offset[7] = self.Chain4_Deadline[6]
        self.Chain4_Deadline[7] = self.Chain4_Offset[7] + self.chain4_buffer[7]
        self.Chain4_Offset[8] = self.Chain4_Deadline[7]
        self.Chain4_Deadline[8] = self.Chain4_Offset[8] + self.chain4_buffer[8]
        self.Chain4_Offset[3] = self.Chain4_Deadline[8]
        self.Chain4_Deadline[3] = self.Chain4_Offset[3] + self.chain4_buffer[3]
        self.Chain4_Offset[4] = self.Chain4_Deadline[3]
        self.Chain4_Deadline[4] = self.Chain4_Offset[4] + self.chain4_buffer[4]
        self.Chain4_Offset[5] = self.Chain4_Deadline[4]
        self.Chain4_Deadline[5] = self.Chain4_Offset[5] + self.chain4_buffer[5]

        # print("Chain4_Check Offset:", self.chain4_buffer)
        # print("Chain4_Check Offset:", self.Chain4_Offset)
        # print("Chain4_Check Offset:", self.Chain4_Deadline)

        # print("End-to-End Deadline Check, Sum of Task WCET in each Path:", sum(self.chain1_buffer))
        # print("End-to-End Deadline Check, Sum of Task WCET in each Path:", sum(self.chain2_buffer))
        # print("End-to-End Deadline Check, Sum of Task WCET in each Path:", sum(self.chain3_buffer))
        # print("End-to-End Deadline Check, Sum of Task WCET in each Path:", sum(self.chain4_buffer))

        # Rule 3 Adaption
        for i in range(len(self._state_rrh)):
            self.Task_Offset[i] = max(self.Chain1_Offset[i], self.Chain2_Offset[i], self.Chain3_Offset[i],
                                      self.Chain4_Offset[i])

            self.Task_Deadline[i] = min(self.Chain1_Deadline[i], self.Chain2_Deadline[i], self.Chain3_Deadline[i],
                                        self.Chain4_Deadline[i])
            self.Task_Offset[i] = round(self.Task_Offset[i], 2)
            self.Task_Deadline[i] = round(self.Task_Deadline[i], 2)

        print("Final Rule #3 Check: offset, Deadline is", self.Task_Offset)
        print("Final Rule #3 Check: offset, Deadline is", self.Task_Deadline)
        # print("Final Rule #3 Check: offset, Deadline is", self.Chain1_Deadline)
        # print("Final Rule #3 Check: offset, Deadline is", self.Chain2_Deadline)
        # print("Final Rule #3 Check: offset, Deadline is", self.Chain3_Deadline)
        # print("Final Rule #3 Check: offset, Deadline is", self.Chain4_Deadline)

        # Reset Value
        for i in range(len(self._state_rrh)):
            self.Chain1_Offset[i] = 0
            self.Chain2_Offset[i] = 0
            self.Chain3_Offset[i] = 0
            self.Chain4_Offset[i] = 0
            self.Chain1_Deadline[i] = 100
            self.Chain2_Deadline[i] = 100
            self.Chain3_Deadline[i] = 100
            self.Chain4_Deadline[i] = 100

    def reset_channel(self):
        self._paras = self._init_channel()
        self._opter = CvxOpt()

    def reset_demand(self):
        self._demand = self._get_demand()
        self._paras['cof'] = self._get_factor(rk_demand=self._demand)

    def reset_state(self):
        #self._state_rrh = np.zeros(self._num_rrh) + self.STATE_ON
        self._state_rrh = np.zeros(self._num_rrh) + np.random.randint(0, self._num_core - 1, size=self._num_rrh)
        print("state_rest:", self._state_rrh)
        self._state_rrh_last = self._state_rrh.copy()

        return self.state

    def step(self, action):
        _, _, _ = self.sub_step(action)
        #power, reward, done = self.perform()
        reward = self._get_max_util_reward()
        #power = self._load_diff
        power = self._load_std
        #power = self.ScheduleOkay/(self.ScheduleError + self.ScheduleOkay)
        done = False

        # done = True if stop else done
        return self.state, power, reward, done

    def sub_step(self, action):
        action_index = np.argmax(action)
        #print("action_index:", action_index)

        if action_index == self.dim_action - 1:
            # stop=True
            return self.state, 0, True

        if self._num_core == 2:
            if action_index % 2 == 1:
                self._state_rrh[int(action_index / 2)] = 1
            elif action_index % 2 == 0:
                self._state_rrh[int(action_index / 2)] = 0

        elif self._num_core == 3:
            if action_index % 3 == 2:
                self._state_rrh[int(action_index / 3)] = 2
            elif action_index % 3 == 1:
                self._state_rrh[int(action_index / 3)] = 1
            elif action_index % 3 == 0:
                self._state_rrh[int(action_index / 3)] = 0

        elif self._num_core == 4:
            if action_index % 4 == 3:
                self._state_rrh[int(action_index / 4)] = 3
            elif action_index % 4 == 2:
                self._state_rrh[int(action_index / 4)] = 2
            elif action_index % 4 == 1:
                self._state_rrh[int(action_index / 4)] = 1
            elif action_index % 4 == 0:
                self._state_rrh[int(action_index / 4)] = 0

        return self.state, 0, False

    def perform(self):
        power, reward, done = self._get_power_reward_done(self._state_rrh, self._state_rrh_last)
        self._state_rrh_last = self._state_rrh.copy()
        return power, reward, done

    def _get_power_reward_done(self, state_rrh, state_last):
        done = False
        solution = self._get_solution(state_rrh)
        if solution:
            power, reward = self._get_reward(solution, state_rrh, state_last)
        else:
            # todo: replace power with a reasonable value, can not be 0
            power = reward = self.REWARD_NEG
            done = True
        return power, reward, done

    def _get_solution(self, state_rrh):
        on_index = np.where(state_rrh == self.STATE_ON)[0].tolist()
        num_on = len(on_index)

        # No active RRH
        if num_on == 0:
            return None

        self._opter.feed(
            h=self._paras['h'][on_index, :],
            cof=self._paras['cof'],
            p=self._paras['pl'][on_index],
            theta=self._paras['theta'],
            num_rrh=num_on,
            num_usr=self._num_usr
        )

        solution = self._opter.solve()

        if solution['x'] is None:
            return None
        else:
            return solution
 
    def _get_reward(self, solution, state_rrh, state_rrh_last):
        num_on = len((np.where(state_rrh == self.STATE_ON))[0])
        num_on_last = len((np.where(state_rrh_last == self.STATE_ON))[0])

        num_off = len(np.where(state_rrh == self.STATE_OFF)[0])

        # transition power
        diff = num_on - num_on_last
        power = self._pow_gap * diff if diff > 0 else 0
        # print('trP:', power)

        # on and sleep power
        p = (num_on * self._pow_on + num_off * self._pow_slp)
        power += p
        # print('ooP:', p, 'On:', num_on)

        # transmit power
        p = sum(solution['x'][1:] ** 2) * (1.0 / self._ETA)
        power += p
        # print('tmP:', p)

        # normalized power
        reward_norm = (power - self._P_MIN) / (self._P_MAX - self._P_MIN)

        # power to reward
        reward_norm = 1 - reward_norm

        # power, reward, done
        return power, reward_norm

    def _get_max_util_reward(self):
        reward_util_norm = (self._load_std - self._load_min) / (self._load_max - self._load_min)
        reward_util_norm = 1 - self._load_std

        reward_window_norm = 0
        self.ScheduleError = 0
        self.ScheduleOkay = 0

        # for i in range(len(self._state_rrh - 2)):
        #     if (self.Task_Deadline[i] - self.Task_Offset[i]) < self.Task_Load[i]:
        #         print("****Schedulability Error!*****")
        #         reward_window_norm -= 1.5
        #         self.ScheduleError += 1
        #     else:
        #         #reward_window_norm += 1 - (self.Task_Load[i]/(self.Task_Deadline[i] - self.Task_Offset[i]))
        #         print("****Schedulability Not Error~*****")
        #         self.ScheduleOkay += 1

        # print("****Window_Norm Value is*****", reward_window_norm)

        reward_norm = reward_util_norm + reward_window_norm
        return reward_norm

    def _get_max_rrh_reward(self):
        power, reward, _ = self._get_power_reward_done(self._state_rrh_max, self._state_rrh_max)
        return self._num_rrh, power, reward

    def _get_min_rrh_reward(self):
        power, reward, _ = self._get_power_reward_done(self._state_rrh_min, self._state_rrh_min_last)
        return self._num_usr, power, reward

    def _get_rnd_rrh_reward(self):
        num_on = len((np.where(self._state_rrh_rd == self.STATE_ON))[0])
        power, reward, _ = self._get_power_reward_done(self._state_rrh_rd, self._state_rrh_rd_last)
        return num_on, power, reward

    def _get_max_rrh_solution(self):
        self._state_rrh_max = np.zeros(self._num_rrh) + self.STATE_ON

    def _get_min_rrh_solution(self):
        # todo: get uniform initializer
        self._state_rrh_min_last = self._state_rrh_min.copy()

        rd_num_on = range(self._num_rrh)
        rd_num_on = np.random.choice(rd_num_on, self._num_usr, replace=False)
        self._state_rrh_min = np.zeros(self._num_rrh)
        self._state_rrh_min[rd_num_on] = self.STATE_ON

    def _get_rnd_rrh_solution(self):
        state_rrh = np.zeros(self._num_rrh)
        for i in range(1, self._num_rrh + 1):
            state_rrh[:i] = self.STATE_ON
            _, _, done = self._get_power_reward_done(state_rrh, self._state_rrh_rd_last)
            if not done:
                break

        self._state_rrh_rd_last = self._state_rrh_rd.copy()
        self._state_rrh_rd = state_rrh.copy()

    def _get_gains(self, num_rrh=0, num_usr=0):
#        d = np.random.uniform(0, 800, size = (num_rrh, num_usr))
#        L = 14.81+3.76* np.log2(d)
#        c = -1 * L / 20
#        antenna_gain = 0.9
#        s = 0.8
#        channel_gains = pow(10, c) * math.sqrt((antenna_gain*s)) * np.random.rayleigh(scale=1.0, size=(num_rrh, num_usr))
        channel_gains = np.random.rayleigh(scale=1.0, size=(num_rrh, num_usr))
        channel_gains = cvx.matrix(channel_gains) * self._CONST  # * 1.345522816371604e-06
        return channel_gains

    def _get_factor(self, rk_demand):
        mu = np.array([self._TM * (2 ** (i / self._BAND) - 1) for i in rk_demand])
        factor = cvx.matrix(np.sqrt(1. + (1. / mu)))
        return factor

    def _get_demand(self):
        rk_demand = self._dm[self._dm_index]
        self._dm_index += 1
        return rk_demand

    def _generate_demand(self):
        rd = np.random.uniform(self._DM_MIN, self._DM_MAX, size=(20000, self._num_usr))
        return rd

    def _get_power_bound(self):
        pow_min = 1 * self._pow_on + (self._num_rrh - 1) * self._pow_slp
        pow_max = self._num_rrh * self._pow_on
        pow_max += self._num_rrh * (1.0 / self._ETA) * self._pow_tsm
        pow_max += self._pow_gap
        return pow_min, pow_max

    def _init_channel(self):
        self._demand = self._get_demand()
        p_max = np.zeros(self._num_rrh) + self._pow_tsm
        theta = np.zeros(self._num_usr) + self._THETA_2

        def _get_pl(p_max):
            pl = cvx.matrix(np.sqrt(p_max), size=(1, len(p_max)))
            return pl

        def _get_theta(theta):
            theta = cvx.matrix(np.sqrt(theta), size=(1, len(theta)))
            return theta

        return {
            'h': self._get_gains(num_rrh=self._num_rrh, num_usr=self._num_usr),
            'cof': self._get_factor(rk_demand=self._demand),
            'pl': _get_pl(p_max=p_max),
            'theta': _get_theta(theta=theta)
        }
