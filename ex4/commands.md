Run this to see the help message
python train.py --help

seed: 408
exp_name: ex4
env_name: CartPole-v0
agent_name: rbf, dnq
env_agent: cartpole_dqn, cartpole_rbf, lunarlander_dqn
run_id: ???
train_episodes: 500
max_episode_steps: 200
test_episodes: 10
random_episodes: 10
glie_b: 50
gamma: 0.99
batch_size: 16
buffer_size: 1000
save_video: false
save_logging: true
save_model: true
silent: false
use_wandb: true
run_suffix: 0

source ./venv/bin/activate
cd ex4

## Task 1.1
source ./venv/bin/activate
cd ex4

python train.py agent_name=rbf env_agent=cartpole_rbf exp_name=ex4_task1_a train_episodes=500
python train.py agent_name=rbf env_agent=cartpole_rbf exp_name=ex4_task1_b train_episodes=500
python test.py save_video=true

## Task 3
source ./venv/bin/activate
cd ex4
python train.py agent_name=dqn env_agent=cartpole_dqn exp_name=ex4_task3_a train_episodes=4000
python train.py agent_name=dqn env_agent=lunarlander_dqn exp_name=ex4_task3_b train_episodes=500

# For testing
python train.py agent_name=dqn env_agent=cartpole_dqn exp_name=ex4_task3_c train_episodes=5 save_model=false save_logging=false save_video=true use_wandb=false
python test.py agent_name=dqn env_agent=cartpole_dqn exp_name=ex4_task3_c save_model=false save_logging=false save_video=true use_wandb=false
