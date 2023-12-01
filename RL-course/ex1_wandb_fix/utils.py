import pickle, os, random, torch
import numpy as np
from collections import defaultdict
import pandas as pd 
import gymnasium as gym
import matplotlib.pyplot as plt

def get_space_dim(space):
    t = type(space)
    if t is gym.spaces.Discrete:
        return space.n
    elif t is gym.spaces.Box:
        return space.shape[0]
    else:
        raise TypeError("Unknown space type:", t)

def load_object(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def save_object(obj, filename): 
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)

class Logger(object):
    def __init__(self,):
        self.metrics = defaultdict(list)
        
    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def save(self, path):
        df = pd.DataFrame.from_dict(self.metrics)
        df.to_csv(f'{path}.csv')
        
def plot_reward(path, env_name):
    df = pd.read_csv(path)
    episodes = df['episodes']
    reward = df['ep_reward']
    plt.figure(figsize=(4.5,3))
    plt.plot(episodes, reward, linewidth=1.2)
    plt.xlabel('Episode', fontweight=10)
    plt.ylabel('Average Reward', fontweight=10)
    plt.title(env_name, fontweight=12)
    plt.plot()