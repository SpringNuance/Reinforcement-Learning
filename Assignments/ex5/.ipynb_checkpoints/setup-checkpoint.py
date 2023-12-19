import yaml
import sys, os
sys.path.insert(0, os.path.abspath(".."))
os.environ["MUJOCO_GL"] = "egl" # for pygame rendering
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import utils as u

from pathlib import Path
work_dir = Path().cwd()/'results'

class Struct:
    def __init__(self, **entries):
        self.entries = entries
        self.__dict__.update(entries)
    
    def __str__(self):
        return str(self.entries)
    
    
def setup(cfg_path, cfg_args={}):

    with open(cfg_path, 'r') as f:
        d = yaml.safe_load(f)
        d.update(cfg_args)
        cfg = Struct(**d)
        
    # Setting library seeds
    if cfg.seed == None:
        seed = np.random.randint(low=1, high=1000)
    else:
        seed = cfg.seed
    
    print("Numpy/Torch/Random Seed: ", seed)
    u.set_seed(seed) # set seed

    # Define a run id based on current time
    cfg.run_id = int(time.time())

    # Create folders if needed
    work_dir = Path().cwd()/'results'
    if cfg.save_model: u.make_dir(work_dir/cfg.env_name/"model")

    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir/cfg.env_name/'model'/f'{cfg.env_name}_params.pt'

    # Create the gym env
    env = gym.make(cfg.env_name, render_mode='rgb_array' if cfg.save_video else None)

    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/cfg.env_name/'video'/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir/cfg.env_name/'video'/'train'
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=cfg.exp_name, disable_logger=True)
    return env, cfg