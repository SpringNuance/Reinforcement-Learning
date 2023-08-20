Run this to see the help message
python train.py --help

seed: 408
exp_name: ex6
run_id: ???
testing: false
model_path: default
save_video: false
save_logging: true
save_model: true
use_wandb: true
silent: false
run_suffix: 0
env_name: InvertedPendulum-v4
agent_name: pg_ac
train_episodes: 2000
gamma: 0.99
lr: 0.0005
batch_size: 100
buffer_size: 1000000.0

source ./venv/bin/activate
cd ex6

## Task 1
source ./venv/bin/activate
cd ex6

python train.py exp_name=ex6_task1 ev_name=InvertedPendulum-v4 agent_name=pg_ac train_episodes=2000

## Task 2
source ./venv/bin/activate
cd ex6

python train.py exp_name=ex6_task2 env_name=HalfCheetah-v4 agent_name=ddpg train_episodes=300 +tau=0.005
