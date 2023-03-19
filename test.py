import imageio
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from train import DQN,E_Greedy_Policy
from JSSP_env import JSSP,Job
# Importing the model (function approximator for Q-table)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = JSSP(num_j = 100)

OBS_SIZE = env.num_m * env.j_types * 2 + env.num_m * 2
ACTION_SIZE = env.num_m + 1

Q_network = DQN(OBS_SIZE, 512, ACTION_SIZE).to(device)
name = 'm5_j6_t0220_batch256_hidden512_gamma0.7_lr0.1'
# # load the model and time matrix
Q_network.load_state_dict(torch.load('./models/model_{}.pth'.format(name)))
Q_network.eval()
# load the time matrix
time_matrix = np.loadtxt('./models/time_matrix_{}.txt'.format(name))
# policy = E_Greedy_Policy(1, decay=0.999, min_epsilon=0.001)

def convert_state(state):
    state_tensor = torch.tensor(state.flatten(), device=device).unsqueeze(0)    
    return state_tensor

# =============================== test ===============================
images = []
env.reset()
env.time_matrix = time_matrix
state = env.state
# env.j_order = order
# print('time matrix\n',env.time_matrix)
while not env.done:
    # if job order is not finished
    if len(env.j_order)>0:
        j = env.j_order[0]
        job = Job(id=j[0],start_time=j[1],
                    type=j[2],max_time=j[3],
                    waiting_time=0,allocated=False)
        env.buffer.append(job)
        # remove first element from j_order
        env.j_order = env.j_order[1:]
    for idx,j in enumerate(env.buffer):
        if env.get_action_space(j.type):
            state_tensor = convert_state(state)
            action = Q_network(state_tensor).max(1)[1].view(1, 1).numpy()[0][0]
            # action = random.choice(env.get_action_space(j.type))
            job = j
####################################################################################
            action = env.num_m
            neural = Q_network(state_tensor)[0].detach().numpy()
            temp = -np.inf
            for k in env.get_action_space(j.type):
                if neural[k] > temp:
                    action = k
                temp = neural[k]
####################################################################################
            env.buffer.pop(idx)
            break
        else:
            # it means don't take any action (machine is not available)
            action = env.num_m
    # print('action_space:', env.get_action_space(job.type))            
    next_state, reward, done = env.step(action,job)
    # print('action:', action, 'job_type:',job.type,'time elapsed:',env.time_elapsed)
    # print('status\n', env.status,np.count_nonzero(env.status))
    # print('total machine time used:',env.total_process_time)
    # print('total machine time:',(env.num_m * env.time_elapsed))
    # print('time elapsed:',env.time_elapsed)
    # if env.done:
    #     next_state = None
    state = next_state
    img = env.display()
    images.append(img)
# print(env.display().shape)
imageio.mimsave(f"./results/{name}.mp4",images,fps = 5)
print('Reward:', env.reward)
print('time elapsed:',env.time_elapsed)
# env.display()
print('Done with the test')