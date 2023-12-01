import pickle, os, random, torch
import numpy as np
from collections import defaultdict
import pandas as pd 



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