import os
import string
from collections import OrderedDict
import shutil
import argparse
import pickle
import subprocess


srun_template = string.Template(
    """\
sbatch --job-name $job_name \\
       --mem 500G \\
       --partition learnfair \\
       --constraint pascal \\
       --gres gpu:2 \\
       --nodes 1 \\
       --ntasks-per-node 1 \\
       --cpus-per-task 80 \\
       --time 2880 \\
       --output $sweep_folder/$job_name/stdout.log \\
       --error $sweep_folder/$job_name/stderr.log \\
       $train_script
"""
)

# note all the models will be in folder ./models/
train_template = string.Template(
    """\
#!/bin/bash
python -u $root/apex/main.py \\
--game $game \\
--save_dir $save_dir \\
--num_thread 80 \\
--num_game_per_thread 20 \\
--num_worker_per_thread 1 \\
"""
)


root = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(root))
print("root:", root)

games = [
    "breakout",
    "chopper_command",
    "ms_pacman",
    "enduro",
    "gopher",
    "name_this_game",
    "star_gunner",
    "qbert",
    "wizard_of_wor",
    "zaxxon",
]

sweep_folder = "sweep/thread_batch"
sweep_folder = os.path.join(root, "apex", sweep_folder)

variables = OrderedDict([("game", games)])


if not os.path.exists(sweep_folder):
    print("make dir:", sweep_folder)
    os.makedirs(sweep_folder)
# copy the the sweep script to the folder
shutil.copy2(os.path.realpath(__file__), sweep_folder)


# create list of args for sweeping
job_names = ["game%s" % game for game in variables["game"]]
sweep_args = [{"root": root, "game": game} for game in variables["game"]]


# generate sweep files (srun, train, eval) for each job
srun_files = []
for job_name, arg in zip(job_names, sweep_args):
    exp_folder = os.path.join(sweep_folder, job_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    # train script
    arg["save_dir"] = exp_folder
    train = train_template.substitute(arg)
    train_file = os.path.join(sweep_folder, job_name, "train.sh")
    with open(train_file, "w") as f:
        f.write(train)
    print(train)

    srun_arg = {
        "sweep_folder": sweep_folder,
        "job_name": job_name,
        "train_script": train_file,
    }
    srun = srun_template.substitute(srun_arg)
    srun_file = os.path.join(sweep_folder, job_name, "srun.sh")
    with open(srun_file, "w") as f:
        f.write(srun)
        srun_files.append(srun_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="analysis")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    if not args.dry:
        for srun_file in srun_files:
            p = subprocess.Popen(["sh", srun_file], cwd=os.path.join(root, "apex"))
