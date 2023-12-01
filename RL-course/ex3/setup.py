import yaml, time
import numpy as np
import gymnasium as gym
import utils as u
from pathlib import Path

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

    run_id = int(time.time())
    
    # create folders if needed
    work_dir = Path().cwd()/'results'/f'{cfg.env_name}'
    u.make_dir(work_dir)
    
    # create env
    env = gym.make(cfg.env_name, render_mode='rgb_array' if cfg.save_video else None)
    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 500
            video_path = work_dir/'video'/'train'
            
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0, # save video every x episode
                                        name_prefix=cfg.exp_name, disable_logger=True)
    return env, cfg