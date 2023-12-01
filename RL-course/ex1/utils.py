import os, random, torch, pickle
import gymnasium as gym
import numpy as np
from collections import defaultdict

def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

def get_space_dim(space):
    t = type(space)
    if t is gym.spaces.Discrete:
        return space.n
    elif t is gym.spaces.Box:
        return space.shape[0]
    else:
        raise TypeError("Unknown space type:", t)
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)

def save_object(obj, filename): 
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class Logger(object):
    def __init__(self,):
        self.metrics = defaultdict(list)
        
    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def save(self, path):
        save_object(self.metrics, path)