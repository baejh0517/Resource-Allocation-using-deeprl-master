import collections
import tflearn
import tensorflow as tf
import numpy as np
from Agent.dqnet import DQNetwork
#from Agent.dqnet import DoubleDQNetwork
from Utils.explorer import Explorer
from Utils.replaybuffer import ReplayBuffer
from Utils.summary import Summary
from Env.env import Env
from parsers import CRANParser
import matplotlib.pyplot as plt

#for data plotting
#----------------------
import math
import time
from statistics import median


sum_total_power = 0
demand_counter = 0
total_power = []
user_demand = []
previous_time = time.time()
slot_power = []
average_slot_power = []

#### Jaehyun Parameter #####
log_util = [0] * 1000
log_std = [0] * 1000
log_x_lim = [0] * 1000
log_index = 0
Index_max = 100
DQN_value = 0
log_index_2 = 0
Total_DQN_value = [0] * 150
Total_Performance_ratio = [0] * 150
Total_Performance_ratio2 = [0] * 150
Total_Performance_average_ratio1 = [0] * 150
Total_Performance_average_ratio2 = [0] * 150
Worst_value = [0] * 150
mean_value = [0] * 150
std_value = [0] * 150
Final_value = [0] * 150
Total_Pfcnt = 0
Average_cnt = 0
#----------------------

class DQNAgent:

    def __init__(self, env, config):
        self._sess = tf.compat.v1.Session()
        self._env = env

        self._dqn = DQNetwork(self._sess, env.dim_state, env.dim_action, config.lr)
 #       self._doubledqn = DoubleDQNetwork(self._sess, env.dim_state, env.dim_action, config.lr)

        self._dir_mod_full = '{0}/{1}-dqn'.format(config.dir_mod, config.run_id)
        dir_sum_full = '{0}/{1}-dqn'.format(config.dir_sum, config.run_id)
        self._dir_log_full = '{0}/{1}-{2}.log'.format(config.dir_log, config.run_id, 'dqn')

        self._summer = Summary(self._sess)
        self._summer.add_writer(dir_sum_full, name="dqn")
   #     self._summer.add_writer(dir_sum_full, name="doubledqn")
        self._summer.add_writer(dir_sum_full + '-max', name="max")
        self._summer.add_writer(dir_sum_full + '-min', name="min")
        self._summer.add_writer(dir_sum_full + '-rnd', name="rnd")
        self._summer.add_variable(name='ep-sum-reward')
        self._summer.add_variable(name='ep-mean-power')
        self._summer.add_variable(name='ep-loss')
        self._summer.add_variable(name='ep-rrh')
        self._summer.build()

        self._f_out = open(self._dir_log_full, 'w')
        self._store_args(config, self._f_out)

        self._replay_buffer = ReplayBuffer(config.buffer_size)
        self._explorer = Explorer(config.epsilon_init, config.epsilon_final, config.epsilon_steps)

        self._saver = tf.compat.v1.train.Saver(max_to_keep=5)

        self._train_flag = not config.load_id
        self._max_test_episodes = config.tests

        if config.load_id:
            self._load(config.dir_mod, config.load_id)
        else:
            self._sess.run(tf.compat.v1.global_variables_initializer())

        tflearn.is_training(self._train_flag, session=self._sess)

        self._OBVS = config.observations
        self._BATCH = config.mini_batch
        self._GAMMA = config.gamma

        self._max_episodes = config.episodes
        self._max_ep_sts = config.epochs
        self._max_steps = config.update

        self._ep = 0
        self._st = 0
        self._save_ep = config.save_ep
        self.reset_log()

    def reset_log(self):

        self._ep_reward = {
            'drl': [],
            'rnd': [],
            'min': [],
            'max': [],
        }

        self._ep_power = {
            'drl': [],
            'rnd': [],
            'min': [],
            'max': [],
        }

        self._ep_maxq = 0
        self._ep_loss = []
        self._actions = []
        self._rnd_ons = []

    def predict(self, state):
        """if(option==1):
            q_value = self._dqn.predict([state])[0]
            self._ep_maxq = np.max(q_value)
            
        if(option==2):
            q_value = self._doubledqn.predict([state])[0]
            self._ep_maxq = np.argmax(q_value)
            
        """
        q_value = self._dqn.predict([state])[0]
        # print("***q_value***:", q_value)
        # print("***state***:", [state])
        self._ep_maxq = np.max(q_value)

        if self._train_flag:
            act = self._explorer.get_action(q_value)
            #print("train_flag act:", act)
        else:
            act = self._explorer.get_pure_action(q_value)
            print("pure act:", act)
        return act
    
    def work(self, num_Avg):
        
        global sum_total_power
        global demand_counter
        sum_total_power = 0
        demand_counter = 0

        init_state = state = self._env.reset_state()
        max_episodes = self._max_episodes if self._train_flag else self._max_test_episodes
        reset_state_ep = self._save_ep if self._train_flag else 20

        for _ in range(max_episodes):

            self._env.reset_demand()
            self._env.run_fix_solution()
            if not self._ep % reset_state_ep:
                init_state = state = self._env.reset_state()

            self._ep += 1
            self._explorer.decay()

            for ep_st in range(self._max_ep_sts):
                self._st += 1

                action = self.predict(state)
                state_next, util, reward, done = self._env.step(action)

                _, power_max, reward_max = self._env.max_rrh_reward
                _, power_min, reward_min = self._env.min_rrh_reward
                on_rnd, power_rnd, reward_rnd = self._env.rnd_rrh_reward

                self._rnd_ons.append(on_rnd)

                self._ep_reward['drl'].append(reward)
                self._ep_reward['rnd'].append(reward_rnd)
                self._ep_reward['max'].append(reward_max)
                self._ep_reward['min'].append(reward_min)

                self._ep_power['drl'].append(util)
                self._ep_power['rnd'].append(power_rnd)
                self._ep_power['max'].append(power_max)
                self._ep_power['min'].append(power_min)

                self._actions.append(np.max(action))

                self._train_batch((state, action, reward, state_next, done))

                state = state_next
                global log_util
                global log_index
                global log_index_2
                global log_x_lim
                global log_std
                global Index_max
                global DQN_value

                log_util[log_index_2] = util
                log_x_lim[log_index_2] = log_index_2
                log_index = log_index + 1

                if log_index % 30 == 0:
                    log_index_2 = log_index_2 + 1
                    if log_index_2 % Index_max == 0:
                        # import pdb
                        # pdb.set_trace()
                        log_util = log_util[:Index_max]
                        log_x_lim = log_x_lim[:Index_max]
                        ######
                        list_log = [log_util, log_x_lim]
                        list_log_np = np.array(list_log)
                        np.save('logging.npy', list_log_np)
                        ######
                        plt.plot(log_x_lim, log_util, 'g', marker='.')
                        plt.xlabel('N_Epochs')
                        plt.ylabel('Standard Deviation(σ)')
                        plt.xlim(0, 99)
                        plt.ylim(0, 1)
                        plt.grid(True)

                        DQN_value = min(log_util)

                        plt
                        plt.show()
                if done:
                    break

            if self._train_flag and not self._ep % self._save_ep:
                self.save()
                self._write_log(init_state, state)
                init_state = state
                self.reset_log()
                
        
        #for data plotting
        #----------------------     
        
        mean_total_power = 0
        
        if (sum_total_power != 0):
            mean_total_power = sum_total_power/demand_counter
            global total_power
            global user_demand

            total_power.append(mean_total_power)
            user_demand.append(self._env._DM_MAX/1.e6)

        self.save_info(user_demand, total_power, average_slot_power)
            #for j in range(len(total_power)):
             #   print ('power', total_power[j])
        #----------------------

        ########################### Jaehyun BAE Code ###############################

        Performance_ratio = self._env.find_optimal_value(DQN_value)
        Performance_ratio2 = self._env.optimal_value/DQN_value

        print("************Final Total Utilzation:", sum(self._env._task_util))
        print("************Final Task Utilzation:", self._env._task_util)
        print("************Final Minimum of DQN Value is:", DQN_value)
        print("************Final Minimum of Optimal Value is:", self._env.optimal_value)

        print("************Final Ratio of Optimal & DQN:", Performance_ratio)

        global Total_DQN_value
        global Total_Performance_ratio
        global Total_Performance_ratio2
        global Total_Performance_average_ratio1
        global Total_Performance_average_ratio2
        global Total_Pfcnt
        global Average_cnt
        global Worst_value
        global mean_value
        global std_value
        global Final_value

        Total_Performance_ratio[Total_Pfcnt] = round(Performance_ratio, 3)     ### Five Slice for each "utilization bound" approach, ex) 0.2 to 0.4, 0.4 to 0.6, ... 
        Total_Performance_ratio2[Total_Pfcnt] = round(Performance_ratio2, 3)
        Total_DQN_value[Total_Pfcnt] = DQN_value

        # if Total_Pfcnt % num_Avg == 2:
        # Total_Performance_average_ratio1[Average_cnt] = round(Total_Performance_ratio[Total_Pfcnt]/num_Avg, 3)
        # Total_Performance_average_ratio2[Average_cnt] = round(Total_Performance_ratio2[Total_Pfcnt]/num_Avg, 3)

        # Worst_value[Average_cnt] = round(self._env.max_value, 3)
        # mean_value[Average_cnt] = round(self._env.mean_value, 3)
        # std_value[Average_cnt] = round(self._env.std_value, 3)

        # _dqn = (Total_DQN_value[Total_Pfcnt-2] + Total_DQN_value[Total_Pfcnt-1] + Total_DQN_value[Total_Pfcnt])/num_Avg
        # _mean = mean_value[Average_cnt]
        # _std = std_value[Average_cnt]
        # _optimal = self._env.optimal_value

        #Final_value[Average_cnt] = round((_dqn - _mean)/_std , 3)

        Total_Performance_average_ratio1[Total_Pfcnt] = round(Total_Performance_ratio[Total_Pfcnt]/num_Avg, 3)
        Total_Performance_average_ratio2[Total_Pfcnt] = round(Total_Performance_ratio2[Total_Pfcnt]/num_Avg, 3)

        Worst_value[Total_Pfcnt] = round(self._env.max_value, 3)
        mean_value[Total_Pfcnt] = round(self._env.mean_value, 3)
        std_value[Total_Pfcnt] = round(self._env.std_value, 3)

        _dqn = Total_DQN_value[Total_Pfcnt]/num_Avg
        _mean = mean_value[Total_Pfcnt]
        _std = std_value[Total_Pfcnt]
        _optimal = self._env.optimal_value
        Final_value[Total_Pfcnt] = round((_dqn - _mean)/(_optimal - _mean), 3)

        #Average_cnt += 1

        Total_Pfcnt += 1
        print("************Final Total_Performance_ratio#1:", Total_Performance_ratio)
        print("************Final Total_Performance_ratio#2:", Total_Performance_ratio2)
        print("************Final Average_Performance_ratio#1:", Total_Performance_average_ratio1)
        print("************Final Average_Performance_ratio#2:", Total_Performance_average_ratio2)

        print("//////////////////////////////////////////")
        print("************Worst Value is:", Worst_value)
        print("************Mean Value is:", mean_value)
        print("************Std Value is:", std_value)
        print("************Final Value is:", Final_value)

        # Global Parameter Reset except related to Total_Performance
        log_util = [0] * 1000
        log_std = [0] * 1000
        log_x_lim = [0] * 1000
        log_index = 0
        Index_max = 100
        DQN_value = 0
        log_index_2 = 0
    

    def _train_batch(self, sample):
        self._replay_buffer.add_samples([sample])

        if len(self._replay_buffer) < self._OBVS or len(self._replay_buffer) < self._BATCH:
            return False

        batch_state, batch_action, batch_reward, batch_state_next, batch_done = \
            self._replay_buffer.sample_batch(self._BATCH)
        q_values = self._dqn.predict_target(batch_state_next)
        # q_values = self._dqn.predict(batch_state_next)
        batch_y = []

        for q, reward, done in zip(q_values, batch_reward, batch_done):
            if done:
                batch_y.append(reward)
            else:
                batch_y.append(reward + self._GAMMA * np.max(q))

        _, loss = self._dqn.train(batch_state, batch_action, batch_y)
        self._ep_loss.append(loss)

        self._dqn.update_target()

        return True

    def _write_log(self, last_state, state):

        total_epochs = len(self._ep_reward['drl'])

        reward = np.array([self._ep_reward['drl'], self._ep_reward['rnd'], self._ep_reward['min'], self._ep_reward['max']])
        power = np.array([self._ep_power['drl'], self._ep_power['rnd'], self._ep_power['min'], self._ep_power['max']])

        index_non_zeros = (power[0, :] != 0) #& (power[1, :] != 0)

        reward = reward[:, index_non_zeros]
        power = power[:, index_non_zeros]
        
         
        total_epochs_non_0 = len(reward[0])

        reward = np.mean(reward, axis=1)
        power = np.mean(power, axis=1)

        reward = {'drl': reward[0], 'rnd': reward[1], 'min': reward[2], 'max': reward[3]}

        power = {'drl': power[0], 'rnd': power[1], 'min': power[2], 'max': power[3]}

        counter = collections.Counter(self._actions)
        init_state = ['{0:.0f}'.format(i) for i in last_state][:self._env.num_rrh]
        final_state = ['{0:.0f}'.format(i) for i in state][:self._env.num_rrh]
        tmp = ' '.join(['| Episode: {0:.0f}'.format(self._ep),
                        '| Demand: {0}'.format(self._env.demand),
                        '| Epsilon: {0:.4f}'.format(self._explorer.epsilon),
                        '| Agent-steps: %i' % self._st,
                        '| Length: before {0} after {1}'.format(total_epochs, total_epochs_non_0),
                        '| Ep-max-reward: {0:.4f}'.format(reward['max']),
                        '| Ep-min-reward: {0:.4f}'.format(reward['min']),
                        '| Ep-rnd-reward: {0:.0f} {1:.4f}'.format(self._rnd_ons[-1], reward['rnd']),
                        '| Ep-reward: {0:.4f}'.format(reward['drl']),
                        '| Ep-max-power: {0:.4f}'.format(power['max']),
                        '| Ep-min-power: {0:.4f}'.format(power['min']),
                        '| Ep-rnd-power: {0:.4f}'.format(power['rnd']),
                        '| Ep-power: {0:.4f}'.format(power['drl']),
                        '| Num-rrh-on: %i' % self._env.num_rrh_on,
                        '| Ep-action: {0}'.format([(k, counter[k]) for k in sorted(counter.keys())]),
                        '| Init-state: {0}'.format('-'.join(init_state)),
                        '| Final-state: {0}'.format('-'.join(final_state))])
        
        #for data plotting
        #----------------------
        
        global slot_power   
        global slot_episode_counter
        global previous_time
        
        global sum_total_power
        global demand_counter
        # global log_util
        # global log_index
        # global log_x_lim
        
        if (math.isnan(power['drl']) == False):
            
            if(len(average_slot_power) < 20):
                slot_power.append(power['drl'])
                current_time = time.time()
                if ((current_time - previous_time) >= 1):
                    average_slot_power.append(median(slot_power))
                    previous_time = time.time()
                    print('saved')
            
            print(power['drl'])
            # log_util[log_index] = round(power['drl'], 3)
            # print("***********Utilization_Diff:", log_util)

            sum_total_power += power['drl']
            demand_counter += 1
            # log_x_lim[log_index] = log_index
            #
            # log_index = log_index + 1
            #
            # if log_index % 10 == 0:
            #     plt.plot(log_x_lim, log_util, 'g', marker='o')
            #     plt.xlabel('N_Epochs(#)')
            #     plt.ylabel('Utilization Difference(%)')
            #     plt.xlim(0, 50)
            #     plt.ylim(0, 1)
            #     plt.grid(True)
            #     plt
            #     plt.show()

            #print( self._env._DM_MAX)
            print('sum_total_power: ', sum_total_power)
            #print('demand_counter: ', demand_counter)
        
        
        
        #----------------------
    
    
        print(tmp)
        self._f_out.write(tmp + '\n')
        self._f_out.flush()

        if len(self._ep_loss) > 0:
            self._summer.run(feed_dict={
                'ep-loss': np.mean(self._ep_loss),
                'ep-rrh': self._env.num_rrh_on,
                'ep-sum-reward': reward['drl'],
                'ep-mean-power': power['drl'],
            }, name='dqn', step=self._ep)
            self._summer.run(feed_dict={
                'ep-sum-reward': reward['max'],
                'ep-mean-power': power['max'],
            }, name='max', step=self._ep)
            self._summer.run(feed_dict={
                'ep-sum-reward': reward['min'],
                'ep-mean-power': power['min'],
            }, name='min', step=self._ep)
            self._summer.run(feed_dict={
                'ep-sum-reward': reward['rnd'],
                'ep-mean-power': power['rnd'],
            }, name='rnd', step=self._ep)

    def save(self):
        save_path = self._saver.save(self._sess, self._dir_mod_full + '/model',
                                     global_step=self._ep, write_meta_graph=False)
        # tf.logging.info("Model saved in file: {0}".format(save_path))

    def _load(self, dir_mod, load_id):
        self._saver.restore(self._sess, tf.train.latest_checkpoint(dir_mod + '/' + load_id))
        tf.logging.info('Model restored from {0}'.format(load_id))

    @staticmethod
    def _store_args(config, f_out):
        tmp = ''
        for k in sorted(config.__dict__.keys()):
            tmp += '{0:<15} : {1}\n'.format(k, config.__dict__[k])
        print(tmp)
        f_out.write(tmp)
        f_out.flush()
        
    def save_info(self, user_demand, total_power, average_slot_power):
        with open("dqn.txt", "w") as f:
            for (user_demand, total_power, average_slot_power) in zip(user_demand, total_power, average_slot_power):
                f.write("{0},{1},{2},\n".format(user_demand, total_power, average_slot_power))
                
def main(args):
    
    d_min, d_max, n_rrh, n_usr, n_epochs = args
    for i in range(len(d_min)):
        
        parser = CRANParser()
        parser.set_defaults(demand_min=d_min[i], demand_max=d_max[i], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
        config = parser.parse_args()
        print("Eenie")
        env = Env('master', config)
        print("Meenie")
        #gains = env._get_gains(n_rrh, n_usr) 
        #print (gains)                        
        #dqn_agent = DQNAgent(env, config) 
        agent =DQNAgent(env, config) 
        #agent = DoubleDQNAgent(env, config)
        print("Minie")                      
        tf.reset_default_graph()              
        agent.work()                        
    


if __name__ == '__main__':
    d_min = [0.e6, 10.e6, 20.e6, 30.e6, 40.e6, 50.e6]   
    d_max = [10.e6, 20.e6, 30.e6, 40.e6, 50.e6, 60.e6]  
    #d_min = [10.e6]                                      
    #d_max = [60.e6]                                      
    n_rrh = 10
    n_usr = 8
    n_epochs = 10
    args = d_min, d_max, n_rrh, n_usr, n_epochs
    
    main(args)
    plt.plot(user_demand, total_power, 'r', linestyle=':', marker='*')
    plt.xlabel('User Demand (Mbps)')
    plt.ylabel('Total Power Consumption (Watts)')
    plt.xlim(10, 60) 
    plt.ylim(min(total_power)-(min(total_power)%5), 5-(max(total_power)%5)+max(total_power))
    banner = 'Scenario:', n_rrh , ' RRHs and' , n_usr , ' users'
    plt.title(banner)
    plt.grid(True)
    plt
    plt.show()
    print(total_power)
    
    parser = CRANParser()
    parser.set_defaults(demand_min=d_min[0], demand_max=d_max[0], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
    config = parser.parse_args()
    print("Eenie")
    env = Env('master', config)
    print("Meenie")
    ch_gain = env._get_gains(n_rrh, n_usr)
    print ('channel gain  = ', ch_gain)
