Run this to see the help message
python train.py --help
exp_name: ex3
seed: 408
env_name: CartPole-v0
train_episodes: 20000
test_episodes: 10
discr: 16
gamma: 0.98
alpha: 0.1
epsilon: glie
glie_b: 0
initial_q: 0.0
save_video: false
save_logging: true
silent: false
use_wandb: true
run_suffix: 0
bool_position: null

source ./venv/bin/activate
cd ex3

## Task 1.1
source ./venv/bin/activate
cd ex3

python train.py exp_name=ex3_task1_1_a epsilon=0.1 train_episodes=20000
python train.py exp_name=ex3_task1_1_b epsilon=glie glie_b=1000 train_episodes=20000
python train.py exp_name=ex3_question1_b epsilon=glie glie_b=1000 train_episodes=1 use_wandb=false
python test.py save_video=true

## Task 1.3
source ./venv/bin/activate
cd ex3
python train.py exp_name=ex3_task1_3_a epsilon=0 initial_q=0 train_episodes=20000 
python train.py exp_name=ex3_task1_3_b epsilon=0 initial_q=50 train_episodes=20000 

# Task 2
source ./venv/bin/activate
cd ex3
python train.py env=lunarlander_v2 exp_name=task2 epsilon=glie glie_b=1000 train_episodes=20000