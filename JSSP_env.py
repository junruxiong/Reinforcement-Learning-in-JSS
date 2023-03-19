import random
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
plt.style.use("seaborn-dark")
plt.rcParams.update({'figure.max_open_warning': 0})
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
# current path
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class JSSP():
    # path = './data/'
    # read data if available
    def __init__(self, num_m = 5, num_j = 100, j_types = 6):
        self.num_m = num_m # number of machines
        self.num_j = num_j # number of jobs
        self.j_types = j_types # number of job types
        # initialize states:
        self.dis_time = np.zeros((self.num_m, self.j_types)) # distributed/allocated time matrix
        self.status = np.full((self.num_m, self.j_types), False, dtype=bool) # machine_job status matrix
        self.processed_time = np.zeros((self.num_m, self.j_types)) # processed time matrix
        self.action_space = np.zeros((self.num_m)) # action space 1: busy, 0: idle
        self.state = np.stack((self.dis_time, self.processed_time)).flatten() # 2D to 3D state matrix
        self.state = np.append(self.state,self.action_space)
        self.state = np.append(self.state, np.zeros(self.num_m))
        # initialize env:
        # self.job = Job # current job
        self.buffer = [] # waiting jobs
        self.time_matrix = self.init_time_matrix()
        self.j_order = self.init_job_order()
        self.total_process_time = 0
        self.reward = 0
        self.time_elapsed = 0
        self.done = False
        self.display_process = []

    def init_time_matrix(self):
        # randomly generate numpy (1,10) array with shape (num_machines,num_jobs), 
        # then randomly mask some of the values to -1
        #? e.g. each column denotes one job type, and each row denotes machine type, 
        #?      and the value denotes a job type operation time to complete the job type 
        #?      corresponding to the machine type, which shows below:
        #?      ([[6.9, 3.3, 6.4, -1, -1, 7.1],
        #?        [8.2, 7.9, 2.2, 4.1, 5.6, 3.8],
        #?        [8.2, 0.3, -1, -1, 8.4, 9.6],
        #?        [6.8, 7.2, 8.3, 2. , 6.8, 1.2],
        #?        [5.2, -1, 2.7, 1.2, -1, 2.9]])
        #? -1 means that the machine is not available for that job type
        # time_matrix = np.random.rand(self.num_m, self.j_types) * 20
        # # up round float to int
        # time_matrix = np.ceil(time_matrix) # np.round(time_matrix, 0)
        time_matrix = np.random.randint(10,30,(self.num_m, self.j_types))
        # randomly mask some of the values to -1
        mask = np.random.rand(self.num_m, self.j_types)
        mask = mask < 0.2
        time_matrix[mask] = -1
        return time_matrix
    
    def init_job_order(self):
        # randomly generate the order of jobs
        #? the example below shows the order of jobs:
        #? columns: job_id, start_time, job_type, max_response_time
        #? 0   0   6   9
        #? 1   4   2   4
        #? …   …   …   …
        #? 9   30  3   7
        j_id = np.arange(self.num_j) 
        # create a random incrementing (1-3) array of s_time
        max_incrementing = 3
        s_time = np.zeros(self.num_j)
        s_time[0] = np.random.randint(0,max_incrementing)
        for i in range(1,self.num_j):
            s_time[i] = s_time[i-1] + np.random.randint(0,max_incrementing)
        # create a list of random job types
        j_type = np.random.randint(0, self.j_types, self.num_j)
        # create a list of random max_time between a range
        max_time = np.random.randint(3, 15, self.num_j)
        # combine j_id, s_time, j_type, max_time to (n,4) array
        job_order = np.array([j_id, s_time, j_type, max_time]).T
        return job_order
        
    def get_action_space(self,job_type):
        # return the action space (idle machine) of current job type
        idx = np.where(self.time_matrix[:, int(job_type)] != -1)[0]
        # return [i for i in idx if not self.status[i].any()]
        status = self.status.copy()
        time_matrix = self.time_matrix.copy()
        time_matrix[time_matrix == -1] = 0
        done_time = self.dis_time + 1 + time_matrix
        status[done_time == (self.time_elapsed+1)] = False
        return [i for i in idx if not status[i].any()]

    def reset(self):
        # self.job = Job # current job
        self.dis_time = np.zeros((self.num_m, self.j_types)) # distribute/allocated time matrix
        self.status = np.full((self.num_m, self.j_types), False, dtype=bool) # machine_job status matrix
        self.processed_time = np.zeros((self.num_m, self.j_types)) # processed time matrix
        self.action_space = np.zeros((self.num_m)) # action space 1: busy, 0: idle
        self.state = np.stack((self.dis_time, self.processed_time)).flatten() # 2D to 3D state matrix
        self.state = np.append(self.state,self.action_space)
        self.state = np.append(self.state, np.zeros(self.num_m))
        self.j_order = self.init_job_order()

        self.buffer = []
        self.total_process_time = 0
        self.reward = 0
        self.time_elapsed = 0
        self.done = False
        self.display_process = []
        return self.state

    #? only input valid action
    def step(self, action, job): # action equavalent to machine id (type)
        # if machine finished jobs, reset processed jobs state to 0
        time_matrix = self.time_matrix.copy()
        time_matrix[time_matrix == -1] = 0
        done_time = self.dis_time + time_matrix
        self.dis_time[done_time == self.time_elapsed] = 0
        self.status[done_time == self.time_elapsed] = False
        self.processed_time[done_time == self.time_elapsed] = 0
        
        # if current machine and job are idle
        if action != self.num_m and \
           np.count_nonzero(self.status[int(action)]) == 0 and \
           self.time_matrix[int(action)][int(job.type)] != -1:
            # update the state
            self.dis_time[action][int(job.type)] = self.time_elapsed # distribute/allocated time
            self.status[action][int(job.type)] = True # machine_job status matrix
            valid = True
            # self.job.allocated = True
        else:
            if job.waiting_time < job.max_time and action != self.num_m:
                self.buffer.append(job)
        # update each job's waiting time in buffer
        for j in self.buffer:
           j.waiting_time += 1 
        self.processed_time[self.status] += 1 # each machine-job processe time + 1
        # update the time elapsed
        self.time_elapsed += 1
        # update the reward
        self.total_process_time += np.count_nonzero(self.status) # count number of True
        self.reward = self.total_process_time/(self.num_m * self.time_elapsed) # machine untilisationppp (current run time/total machine time)
        self.action_space = np.ones((self.num_m)) # action space
        self.action_space[self.get_action_space(job.type)] = 0 # idle machine
        self.state = np.stack((self.dis_time, self.processed_time)).flatten()
        self.state = np.append(self.state,self.action_space)
        self.state = np.append(self.state, self.time_matrix[:,int(job.type)])
        #? state example:
        #? distribute/allocated time matrix(col: job type, row: machine type)
        #? [[0.  0.  0.  0.  0.  0. ]
        #?  [0.  4.  0.  1.  3.  0. ]
        #?  [0.  0.  0.  0.  0.  0. ]
        #?  [0.  0.  0.  0.  0.  0. ]
        #?  [0.  0.  0.  0.  0.  0. ]]
        #?      concatenate(stack)
        #? processed time matrix(col: job type, row: machine type)
        #? [[0.  0.  0.  0.  0.  0. ]
        #?  [0.  1.  0.  4.  5.  0. ]
        #?  [0.  0.  0.  0.  0.  0. ]
        #?  [0.  0.  0.  0.  0.  0. ]
        #?  [0.  0.  0.  0.  0.  0. ]]
        self.display_process.append(self.processed_time.copy())
        # check if the episode is done
        if len(self.j_order)==0 and len(self.buffer)==0:
            self.done = True
        else:
            self.done = False
        return self.state, self.reward, self.done

    def display(self,return_img = True):
        # plot grantt chart
        display_data = {'m':[],'j':[],'s_t':[],'e_t':[],'duration':[],'color':[]}
        # generate random colors
        c_dict = {}
        colors = ['red','green','blue','yellow','orange','purple','pink','brown','gray','cyan']
        for i in range(self.j_types):
            c_dict[i] = colors[i]

        display_process = np.array(self.display_process)
        for m in range(self.num_m):
            for j in range(self.j_types):
                m_j_temp = display_process[:,m,j]
                # remove 0 values
                m_j = m_j_temp[m_j_temp!=0]
                # create index array
                m_j_idx = np.arange(len(m_j_temp))[m_j_temp!=0]
                # split any non-consecutively increasing values in m_i and mapping to index array
                # m_j_split = np.split(m_j, np.where(np.diff(m_j) != 1)[0] + 1)
                m_j_idx_split = np.split(m_j_idx, np.where(np.diff(m_j) != 1)[0] + 1)
                # print(m,j,m_j_idx_split)
                if m_j_idx_split[0].any():
                    for i in m_j_idx_split:
                        display_data['m'].append(m)
                        display_data['j'].append(j)
                        display_data['s_t'].append(i[0])
                        display_data['e_t'].append(i[-1])
                        display_data['duration'].append(i[-1]-i[0]+1)
                        display_data['color'].append(c_dict[j])

        fig, ax = plt.subplots(figsize=(16,6))
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        ax.barh(display_data['m'], display_data['duration'], 
                left=display_data['s_t'],color=display_data['color'])
        # add grid line background with every 5 unit
        ax.set_yticks(np.arange(self.num_m))
        ax.set_xticks(np.arange(0,self.time_elapsed+1,10))
        ax.grid(color='gray', linestyle='dotted',axis='x', linewidth=1, alpha=0.5)

        if return_img:
            #Image from plot
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return data
        else:
            plt.show()
            return display_data


# arriving job
class Job():
    def __init__(self,id=None,start_time=None,type=None,
                 max_time=None,waiting_time=0,allocated=False):
        self.id = id
        self.start_time = start_time
        self.type = type
        self.max_time = max_time
        self.waiting_time = waiting_time
        self.allocated = allocated
    