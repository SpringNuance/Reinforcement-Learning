import time, yaml#, wandb
import gymnasium as gym
import numpy as np
import pickle 
from matplotlib import pyplot as plt
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import utils as u
from buffer import ReplayBuffer

import time

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
    
    cfg.run_id = int(time.time())
    
    # use wandb to store stats
    
    work_dir = Path().cwd()/'results'/cfg.env_name
    # create env
    env = gym.make(cfg.env_name, render_mode='rgb_array' if cfg.save_video else None, max_episode_steps=cfg.max_episode_steps)
    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 100
            video_path = work_dir/'video'/'train'
            
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0, # save video every x episode
                                        name_prefix=cfg.exp_name+'_'+cfg.agent_name, disable_logger=True)
        
    # get number of actions and state dimensions
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    
    return env, cfg

def train(agent, cfg_path, cfg_args={}):
    
    env, cfg = setup(cfg_path, cfg_args=cfg_args)
    L = u.Logger() # create a simple logger to record stats
    # create folders if needed
    work_dir = Path().cwd()/'results'/cfg.env_name
    if cfg.save_logging:
        logging_path = work_dir / 'logging'
        u.make_dir(logging_path)
    if cfg.save_model:
        model_path = work_dir / 'model'
        u.make_dir(model_path)
        
    #  init buffer
    state_shape = env.observation_space.shape
    buffer = ReplayBuffer(state_shape, action_dim=1, max_size=int(cfg.buffer_size))
    
    ts_times = []
    ep_times = []
    update_times = []
    
    train_logs = []
    
    for ep in range(cfg.train_episodes):
        s_ep = time.perf_counter()
        (state, _), done, ep_reward, env_step = env.reset(), False, 0, 0
        eps = max(cfg.glie_b/(cfg.glie_b + ep), 0.05)
        
        # collecting data and fed into replay buffer
        while not done:
            s_ts = time.perf_counter()
            env_step += 1
            if ep < cfg.random_episodes: # in the first #random_episodes, collect random trajectories
                action = env.action_space.sample()
            else:
                # Select and perform an action
                action = agent.get_action(state, eps)
                if isinstance(action, np.ndarray): action = action.item()
                
            next_state, reward, done, _, _ = env.step(action)
            ep_reward += reward

            # Store the transition in replay buffer
            buffer.add(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state
        
            # Perform one update_per_episode step of the optimization
            if ep >= cfg.random_episodes:
                update_info = agent.update(buffer)
                #update_times.append(update_info['update_time'])
            else: update_info = {}

            if env_step >= env._max_episode_steps:
                done = True
            
            e_ts = time.perf_counter()
            ts_times.append(e_ts-s_ts)
            
        e_ep = time.perf_counter()
        ep_times.append(e_ep-s_ep)
        
        #update_avr = np.mean(update_times)
        ep_avr = np.mean(ep_times)
        ts_avr = np.mean(ts_times)
        
        info = {'ep_reward': ep_reward, 'episode': ep, 'epsilon': eps, 'ep_avr' : ep_avr, 'ts_avr': ts_avr}
        #'update_avr' : update_avr
        info.update(update_info)

        
        if cfg.save_logging: L.log(**info)
        #if cfg.save_logging:
        #    L.save(logging_path/'logging.pkl')
        # save model and logging    
        if cfg.save_model:
            agent.save(model_path)
    
        if (not cfg.silent) and (ep % 100 == 0): 
            print(info)
            ep_times = []
            ts_times = []
            update_times = []

    # save model and logging    
    if cfg.save_model:
        agent.save(model_path)
    if cfg.save_logging:
        L.save(logging_path/'logging.pkl')
    
    print('------ Training Finished ------')


def test(agent, cfg_path, cfg_args={}):
    cfg_args.update({'testing':True})
    env, cfg = setup(cfg_path, cfg_args=cfg_args)
    work_dir = Path().cwd()/'results'/cfg.env_name
    model_path = work_dir / 'model'
    # Load policy / q functions
    agent.load(model_path)

    for ep in range(cfg.test_episodes):
        (state, _), done, ep_reward, env_step = env.reset(), False, 0, 0
        rewards = []

        # collecting data and fed into replay buffer
        while not done:
            # Select and perform an action
            action = agent.get_action(state, epsilon=0.0)
            if isinstance(action, np.ndarray): action = action.item()
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
            rewards.append(reward)
            env_step += 1
            if env_step >= cfg.max_episode_steps:
                done = True
        info = {'episode': ep, 'ep_reward': ep_reward}
        if (not cfg.silent): print(info)

               
def plot(cfg_path, cfg_args={}, save_name=None):
    env, cfg = setup(cfg_path, cfg_args=cfg_args)

    # create folders if needed
    work_dir = Path().cwd()/'results'/cfg.env_name
    logging_path = work_dir / 'logging' / 'logging.pkl'
    plot_path = work_dir / save_name
    
    log_data = u.load_object(logging_path)
    
    
    Rs = log_data['ep_reward']
    Eps = log_data['episode']
    
    
    # Plotting the training loss against episode numbers
    plt.plot(Eps, Rs)
    plt.xlabel('Episode Number')
    plt.ylabel('Returns')
    plt.title('Task returns over Episodes')
    plt.grid(True)
    plt.savefig(plot_path)
    plt.show()