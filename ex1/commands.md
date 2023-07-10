Run this to see the help message
python train.py --help

== Configuration groups ==
Compose your configuration from those groups (group=option)

env: *cartpole_v0*, *reacher_v1*

== Config ==
Override anything in the config (foo.bar=value)

These are the flags for python train.py. 
exp_name: ex1
seed: 408
env_name: CartPole-v0
max_episode_steps: 100
train_episodes: 500
batch_size: 64
min_update_samples: 2000
testing: false
model_path: default
save_video: true
save_model: true
save_logging: true
silent: false
use_wandb: true
run_suffix: 0

Commands for assignment 1

After training, please renaming the Cartpole-v0_params.pt file in the model folder to the corresponding task name.

## Task 1
source ./venv/bin/activate
cd ex1

python train.py exp_name=ex1 testing=false seed=1 max_episode_steps=100 use_wandb=True
python train.py exp_name=ex1 model_path=results/model/Task1_params.pt testing=true seed=1 max_episode_steps=1000 use_wandb=false

## Question 1.1
source ./venv/bin/activate
cd ex1

python train.py exp_name=ex111 model_path=results/model/Task1_params.pt testing=true seed=100 max_episode_steps=1000 use_wandb=false

python train.py exp_name=ex112 model_path=results/model/Task1_params.pt testing=true seed=200 max_episode_steps=1000 use_wandb=false

python train.py exp_name=ex113 model_path=results/model/Task1_params.pt testing=true seed=300 max_episode_steps=1000 use_wandb=false

python train.py exp_name=ex114 model_path=results/model/Task1_params.pt testing=true seed=400 max_episode_steps=1000 use_wandb=false

python train.py exp_name=ex115 model_path=results/model/Task1_params.pt testing=true seed=500 max_episode_steps=1000 use_wandb=false

## Task 2:
source ./venv/bin/activate
cd ex1

python train.py exp_name=ex21 testing=false seed=100 max_episode_steps=100 use_wandb=false
python train.py model_path=results/model/Task21_params.pt exp_name=ex21 testing=true seed=100 max_episode_steps=1000 use_wandb=false

python train.py exp_name=ex22 testing=false seed=200 max_episode_steps=100 use_wandb=false
python train.py model_path=results/model/Task22_params.pt exp_name=ex22 testing=true seed=200 max_episode_steps=1000 use_wandb=false

python train.py exp_name=ex23 testing=false seed=300 max_episode_steps=100 use_wandb=false
python train.py model_path=results/model/Task23_params.pt exp_name=ex23 testing=true seed=300 max_episode_steps=1000 use_wandb=false

python train.py exp_name=ex24 testing=false seed=400 max_episode_steps=100 use_wandb=false
python train.py model_path=results/model/Task24_params.pt exp_name=ex24 testing=true seed=400 max_episode_steps=1000 use_wandb=false

python train.py exp_name=ex25 testing=false seed=500 max_episode_steps=100 use_wandb=false
python train.py model_path=results/model/Task25_params.pt exp_name=ex25 testing=true seed=500 max_episode_steps=1000 use_wandb=false

## Task 3:
source ./venv/bin/activate
cd ex1

In reacher.py, change the get_reward function to support each corresponding task

After training, please renaming the Reacher-v1_params.pt file in the model folder to the corresponding task name.

# For training continuous clockwise rotation wrt angle theta_0
python train.py env=reacher_v1 train_episodes=200

# For training to reach the goal point located in x = [1.0,1.0] (marked in red)
python train.py env=reacher_v1 train_episodes=1000