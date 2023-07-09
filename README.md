# rl_course

This is the exercise code structure for the course ELEC-E8125 Reinforcement Learning, 2022.

## Code Structure
When the new exercise is released, you should download it from MyCourses and put it under the `rl_course` folder. The code is given in the following structure
```
rl_course
|__ common
|__ ex1
    |__ cfg
        |__ env
        |__ ex1_cfg.yaml
    |__ train.py
    |__ agent.py
    |__ reacher.py
    |__ plot_rew.ipynb
|__ ex2
...
```

The `common` folder includes a set of helper functions/classes that are useful in exericses. Usually, you don't need to dig this folder, but it will help you have have a better understanding of details.
We use [hydra](https://hydra.cc/) to manage the configuration for each exercise, and all parameters are located under the `cfg` folder. We use [wandb](https://docs.wandb.ai/) to manage experiments. Furthermore, we use jupyter notebook for analysis and visualization.

## How to develop remotely
We recommend write the code using VS Code and run everything on a remote machine offered by Aalto University (the full list is [here](https://www.aalto.fi/en/services/linux-computer-names-in-it-classrooms)), especially if you are using a non-GPU or a Windows machine. For Mac or Linux users, the setting of SSH and VS Code can be found [here](https://yi-zhao.notion.site/How-to-set-up-a-remote-machine-d31e799fe2014fc7b806a29f54c71f1b). For Windows users, this [link](https://code.visualstudio.com/docs/remote/wsl) might be useful. You should be able to open and run `.ipynb` (for Jupyter Notebook) out of box, but here is the [details](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

## How to run the code
Suppose you have successfully set up the VS Code and can connect to a remote GPU machine, and create the vitrual environment. You can run the code by:
1. `cd` to the `rl_course` folder and activate the created vritual enviroment by `source ./venv/bin/activate` (use `deactivate` to deactivate the environment).
2. Wandb login: `wandb login`
3. Run the code: `cd` to the corresponding exercise foloder, such as `cd ex1`, and run, e.g. `python3 train.py save_video=true save_logging=true use_wandb=true`

## How to export plot from wandb
Export a plot from wandb is easy:

1. Open your wandb workspace and select the runs to plot.
2. Edit the plot (e.g., episode-ep_reward plot) you gonna export (click the ✏️ button at the upper right corner of the panel).
    1. In `Data`, change the x-, and y-axis to `episodes` and `ep_reward` 
    2. In `Grouping`, group runs and select `mean` and `stddev` if needed.
    3. In `Chart`, edit the name of the x- and y-axis.
    4. In `Legend`, change the legend name.
3. Export the panel as `.pdf` and include it in your report.

## How to submit
Exercises must be done **individually**. We will use **TurnItIn** to verify this. You can work with peers on the
solutions but the solution you provide must be your own work. It is advised to ask questions in the [Slack channel](https://join.slack.com/t/elece8125rein-0pe1068/shared_invite/zt-1fjpcfg12-UY9UNhgFOF8GxiGp7~olew) and attend to the exercise
sessions as TAs will assist you on the solutions. It's recommended to use the **Slack channel** as the main place to ask questions since other people might ask the same questions. But **avoid sharing your code** in the channel, if the code sharing is needed for your questions, you can have a private discussion with TAs or attend the exercise session.

A good exercise submission report should meet the following criterias:
- The [latex template](https://mycourses.aalto.fi/course/view.php?id=37149&section=2) is used
- Sections in the latex report match the tasks/questions in the instructions
- Plots have axis labels and titles and are included in the report
- Submission are **not** in a compressed zip file, including only the required files
- Please do **not** return the unrequired logging, videos ect.
We will subtract one point per criteria that is not met, up to 5 points total for each exercise.

### How to answer questions
**Tasks**: require you to complete a programming assignment. **The relevant lines you changed or filled in the provided code must be in the report**. **Plots generated during the training of a task must be in the report.**

**Questions**: are divided into:
- Yes/no questions, e.g. Can the provided model be trained without further modification?
- Discussing questions, requires you to briefly explain or justify your answer, e.g. why/why not?

It is usually marked which kind of answer we expect from a question. The questions that require you to justify your answer must be properly argued. Please avoid answers that do not provide any reasoning. Some
examples of answers that are poorly justified and why they are poorly justified are:
- Why does the reinforcement learning algorithm perform better when adding the parameter $\gamma$? Because it takes less time than before. - This answer does not provide any reasoning about how modifying $\gamma$
affects the learning algorithm.
- How does the reward change over multiple runs of the algorithm? The reward is the same at the end of the training. - This answer might be obvious. What is interesting about how it changed, what could be the reason behind it?
