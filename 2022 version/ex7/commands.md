source ./venv/bin/activate
cd ex7

## Task 1
source ./venv/bin/activate
cd ex7

python cem.py -use_wandb=1 # -use_wandb=1 to use wandb
python cem.py -use_wandb=0 # -use_wandb=0 to not use wandb

## Task 2
source ./venv/bin/activate
cd ex7

Go to while node.children: line and change the search policy accordingly 

python az.py -use_wandb=1 # -use_wandb=1 to use wandb
python az.py -use_wandb=0 # -use_wandb=0 to not use wandb

If results do not work as expected, consider erasing common/__pycache__