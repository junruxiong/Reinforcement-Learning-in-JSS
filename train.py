import math
import random 
import numpy as np
from collections import namedtuple, deque 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from JSSP_env import JSSP,Job
# Importing the model (function approximator for Q-table)
# from model import ReplayMemory,Transition

from collections import namedtuple

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def convert_state(state):
    state_tensor = torch.tensor(state.flatten(), device=device).unsqueeze(0)    
    return state_tensor
    
class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        state_tensor = convert_state(state)
        
        if next_state is None:
            state_tensor_next = None            
        else:
            state_tensor_next = convert_state(next_state)
            
        action_tensor = torch.tensor([action], device=device).unsqueeze(0)

        reward = torch.tensor([reward], device=device).unsqueeze(0)/10. # reward scaling

        self.memory[self.position] = Transition(state_tensor, action_tensor, state_tensor_next, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# define neural network
class DQN(nn.Module):

    def __init__(self, input_size, size_hidden, output_size):
        
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, size_hidden)
        self.bn1 = nn.BatchNorm1d(size_hidden)
        
        self.fc2 = nn.Linear(size_hidden, size_hidden)   
        self.bn2 = nn.BatchNorm1d(size_hidden)

        self.fc3 = nn.Linear(size_hidden, size_hidden)  
        self.bn3 = nn.BatchNorm1d(size_hidden)

        self.fc4 = nn.Linear(size_hidden, output_size)
        
        
    def forward(self, x):
        h1 = F.relu(self.bn1(self.fc1(x.float())))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        h3 = F.relu(self.bn3(self.fc3(h2)))
        output = self.fc4(h3.view(h3.size(0), -1))
        return output

# define e-greedy policy
class E_Greedy_Policy():
    
    def __init__(self, epsilon, decay, min_epsilon):
        
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.decay = decay
        self.epsilon_min = min_epsilon
                
    def __call__(self, state, action_space):
        is_greedy = random.random() > self.epsilon
        if is_greedy:
            # we select greedy action
            with torch.no_grad():
                Q_network.eval()
                index_action = Q_network(state).max(1)[1].view(1, 1).numpy()[0][0]
                index_action = env.num_m
                neural = Q_network(state)[0].numpy()
                temp = -1
                for i in action_space:
                    if neural[i] > temp:
                        index_action = i
                    temp = neural[i]
                Q_network.train()
        else:
            # we sample a random action
            # index_action = random.randint(0,5) #! action space
            index_action = random.choice(action_space)
        return index_action
                
    def update_epsilon(self):
        
        self.epsilon = self.epsilon*self.decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        
    def reset(self):
        self.epsilon = self.epsilon_start

def optimize_model():
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute Q values using policy net
    Q_values = Q_network(state_batch).gather(1, action_batch)

    # Compute next Q values using Q_targets
    next_Q_values = torch.zeros( BATCH_SIZE, device=device)
    next_Q_values[non_final_mask] = Q_target(non_final_next_states).max(1)[0].detach()
    next_Q_values = next_Q_values.unsqueeze(1)
    
    # Compute targets
    target_Q_values = (next_Q_values * GAMMA) + reward_batch
    
    # Compute MSE Loss
    loss = F.mse_loss(Q_values, target_Q_values)
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # Trick: gradient clipping
    for param in Q_network.parameters():
        param.grad.data.clamp_(-1, 1)
        
    optimizer.step()
    
    return loss
        
if __name__ == "__main__":
    # =====================================Initialize========================================= #
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    # if gpu is to be used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = JSSP()

    job = Job
    OBS_SIZE = env.num_m * env.j_types * 2 + env.num_m * 2
    HIDDEN_SIZE = 512
    ACTION_SIZE = env.num_m+1
    BATCH_SIZE = 256
    GAMMA = 0.7

    Q_network = DQN(OBS_SIZE, HIDDEN_SIZE, ACTION_SIZE).to(device)
    Q_target = DQN(OBS_SIZE, HIDDEN_SIZE, ACTION_SIZE).to(device)
    Q_target.load_state_dict(Q_network.state_dict())
    Q_target.eval()

    TARGET_UPDATE = 100

    optimizer = optim.SGD(Q_network.parameters(), lr=0.1)
    memory = ReplayMemory(10000)

    policy = E_Greedy_Policy(0.1, decay=0.999, min_epsilon=0.001)

    # ========================================Train========================================= #
    num_episodes = 500
    policy.reset()
    rewards_history = []
    # Warmup phase!
    memory_filled = False
    order = env.init_job_order()

    while not memory_filled:
        env.reset()
        # env.j_order = order
        state = env.state
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
            for idx,i in enumerate(env.buffer):
                if env.get_action_space(i.type):
                    job = i
                    state_tensor = convert_state(state)
                    action = policy(state_tensor, env.get_action_space(job.type))
                    env.buffer.pop(idx)
                    break
                else:
                    # it means don't take any action (machine is not available)
                    action = env.num_m            
            next_state, reward, done = env.step(action,job)
            # print('done',env.done,env.j_order)
            # print('action:', action, 'job_type:',job.type)
            # print('status\n', env.status,np.count_nonzero(env.status))
            # # print('action_space:', env.get_action_space(env.job.type))
            # print('total time used:',env.total_process_time)
            # print('total time:',(env.num_m * env.time_elapsed))
            if env.done:
                next_state = None
            memory.push(state, action, next_state, reward)
            state = next_state
        print(env.reward)
        memory_filled = memory.capacity == len(memory)
    print('Done with the warmup')

    # Training loop
    for i_episode in range(num_episodes):
        env.reset()
        # env.j_order = order
        state = env.state
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
            for idx,i in enumerate(env.buffer):
                if env.get_action_space(i.type):
                    job = i      
                    state_tensor = convert_state(state)
                    action = policy(state_tensor, env.get_action_space(job.type))
                    env.buffer.pop(idx)
                    break
                else:
                    # it means don't take any action (machine is not available)
                    action = env.num_m            
            next_state, reward, done = env.step(action,job)
            # print('action:', action, 'job_type:',job.type)
            # print('status\n', env.status,np.count_nonzero(env.status))
            # # print('action_space:', env.get_action_space(env.job.type))
            # print('total time used:',env.total_process_time)
            # print('total time:',(env.num_m * env.time_elapsed))
            if env.done:
                next_state = None
            memory.push(state, action, next_state, reward)
            state = next_state
            started_training = True
            l = optimize_model()
        policy.update_epsilon()
        rewards_history.append(float(env.reward))
        print('episode_reward',i_episode,env.reward)
        print('time elapsed:',env.time_elapsed)
    
    # save the model
    name = 'm5_j6_t0220_batch256_hidden512_gamma0.7_lr0.1'
    torch.save(Q_network.state_dict(), './models/model_{}.pth'.format(name))
    # save time matrix
    np.savetxt('./models/time_matrix_{}.txt'.format(name),env.time_matrix,fmt='%d')
    print('model and time matrix saved')

    # print(rewards_history)
    plt.plot(rewards_history)
    plt.show()
    