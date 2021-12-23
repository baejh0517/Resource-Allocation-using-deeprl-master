from Agent.dqn import DQNAgent, user_demand, total_power, average_slot_power
#from Agent.duelingdqn import DuelingDQNAgent, user_demand, total_power, average_slot_power
#from Agent.doubledqn import DoubleDQNAgent, user_demand, total_power, average_slot_power
from Env.env import Env
from parsers import CRANParser
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

Num_Task = 0
Num_Util = 0
Num_Avg = 0

def main(args):
    
    d_min, d_max, n_rrh, n_usr, n_epochs = args
    global Num_Task
    global Num_Util
    global Num_Avg

    for j in range(Num_Task):  # Number of task number
        for i in range(Num_Util):  # Number of total Utilization
            for k in range(Num_Avg):  # Number of Average Value
                # print(Num_Task,Num_Util,j,i,k)
                parser = CRANParser()
                parser.set_defaults(demand_min=d_min[i], demand_max=d_max[i], num_rrh=n_rrh[j], num_usr=n_usr, epochs=n_epochs)
                config = parser.parse_args()
                print("why")
                env = Env('master', config)
                dqn_agent = DQNAgent(env, config)
                # dueling_agent =DuelingDQNAgent(env, config)
                # agent = DoubleDQNAgent(env, config)

                tf.compat.v1.reset_default_graph()
                dqn_agent.work(Num_Avg)
        

if __name__ == '__main__':

    ### Hyper-Parameter Setting JaeHyun Project ###
    # u_min = [0.0, 0.5, 1.0, 1.5]   # Slice of Utilization scope min value
    # u_max = [0.5, 1.0, 1.5, 2.0]   # Slice of Utilization scope max value

    u_min = [3.5]   # Slice of Utilization scope min value
    u_max = [4.0]   # Slice of Utilization scope max value
    n_rrh = [15]                                 # Total Task Number

    Num_Task = len(n_rrh)
    Num_Util = len(u_min)
    Num_Avg = 1

    n_usr = 6   # Don't Care...,
    n_epochs = 30
    args = u_min, u_max, n_rrh, n_usr, n_epochs
    
    main(args)
