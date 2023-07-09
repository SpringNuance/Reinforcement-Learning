import sys
sys.path.append("..")
from typing import Sequence, Tuple, Dict, Callable, List
from functools import partial
import copy
from joblib import Parallel, delayed
import numpy as np
import torch
import gym

from common.env import make_env
from common.simulator import SimulatorWrapper


class CEM(object):
    def __init__(self, 
                model,
                action_shape,
                num_samples,
                num_topk,
                plan_horizon,
                iteration,
                keep_last_solution,
                momentum,
                expl_noise,
                ):

        self.model = model # the dynamic model

        self.action_dim = action_shape[0]
        self.num_samples = num_samples
        self.num_topk = num_topk
        self.plan_horizon = plan_horizon
        self.iteration = iteration
        self.keep_last_solution = keep_last_solution
        self.momentum = momentum
        self.expl_noise = expl_noise

        # init simulator
        o = self.model.reset()
        self.model.save_checkpoint()

        
    def plan(self, obs, t0, eval_mode=False):
        if obs.ndim == 1: obs = obs[None] # add batch dim
        # initialize paramters
        mean = np.zeros((self.plan_horizon, self.action_dim))
        std = np.ones_like(mean)
        # use previous plan as start point if not at the first step
        if not t0 and hasattr(self, "_prev_mean"):
            mean[:-1] = copy.copy(self._prev_mean[1:])

        with Parallel(n_jobs=-1,) as parallel:  # we use joblib.Parallel to parallel the evaluation.
            # Iterate CEM
            for _ in range(self.iteration):
                # TODO: Task 1 Implement CEM
                ########## Your code starts here. ##########
                # Hints: 
                # 1. select actions, note plan horizon and number of samples and action dimensionality
                # 2. evaluate actions by computing values for your actions
                # use parallel(delayed(rollout_simulator)(model, action) for each sample 
                # 3. select top actions (elite actions) in samples (highest returns)
                # 4. compute new mean and std, note that we used momentum for mean

                _mean, _std = None, None # change this line 
                mean, std = self.momentum * mean + (1.0 - self.momentum) * _mean, _std


        if self.keep_last_solution:
            self._prev_mean = mean

        # select the first action in the planed horizon
        action, std = mean[0], std[0]

        if not eval_mode:
            action += self.expl_noise * np.random.randn(action.shape)

        # udpate the simulator state (if use simulator to do planning)
        o, r, d, _ = self.model.step(action)
        self.model.save_checkpoint()

        return action   
        

def rollout_simulator(model, traj):
    model.load_checkpoint()

    terminated, G = False, 0
    for act in traj:
        obs, reward, done, _ = model.step(act)
        reward = 0 if terminated else reward

        terminated |= bool(done)
        if done: break

        G += reward
    return G


#%%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-use_wandb', type=int, default=1)
    args = parser.parse_args()    
    if args.use_wandb:
        import wandb
        import uuid
        run_id = str(uuid.uuid4())
        wandb.init(project="rl_aalto",
                    name=f'ex7-CupCatch-{run_id}',
                    group=f'ex7-CupCatch')

    env = make_env(env_name='cup-catch',
                    seed=1,
                    action_repeat=6)
    obs_shape = tuple(int(x) for x in env.observation_space.shape)
    action_shape = tuple(int(x)  for x in env.action_space.shape)

    model = SimulatorWrapper(env)

    agent = CEM(model=model,
                action_shape = action_shape,
                num_samples=50,
                num_topk=5,
                plan_horizon=12,
                iteration=5,
                keep_last_solution=True,
                momentum=0.1,
                expl_noise=0.3,)

    # %%
    obs, done, ep_reward, t = env.reset(), False, 0, 0

    while not done:
        action = agent.plan(obs, eval_mode=True, t0=(t==0))

        obs, reward, done, _ = env.step(action)
        print(reward)
        ep_reward += reward

        t += 1
        if args.use_wandb:
            wandb.log({
                'Step': t,
                'Reward': reward,
                'Episode Reward': ep_reward
            })
# %%
