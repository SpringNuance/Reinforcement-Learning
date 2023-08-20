Run this to see the help message
python train.py --help

seed: 408
exp_name: ex5
env_name: InvertedPendulum-v4
run_id: ???
train_episodes: 1000
gamma: 0.99
lr: 0.002
testing: false
model_path: default
save_video: false
save_logging: true
save_model: true
use_wandb: true
silent: false
run_suffix: 0

source ./venv/bin/activate
cd ex5

## Task 1
source ./venv/bin/activate
cd ex5

Go to agent.py and change the options accordingly for each Task in two functions
def __init__ of Policy class and def update in PG class
(a) python train.py exp_name=ex5_task1_a train_episodes=1000
(b) python train.py exp_name=ex5_task1_b train_episodes=1000
(c) python train.py exp_name=ex5_task1_c train_episodes=1000

## Task 2
Go to agent.py and change the options accordingly for Task 2 in two functions
def __init__ of Policy class and def update in PG class

source ./venv/bin/activate
cd ex5
python train.py exp_name=ex5_task2 train_episodes=1000
