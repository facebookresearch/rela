import os
import string
from collections import OrderedDict
import shutil
import argparse
import pickle
import subprocess


srun_template = string.Template(
"""\
sbatch --job-name $JOB_NAME \\
       --mem 500G \\
       --partition learnfair \\
       --constraint pascal \\
       --gres gpu:2 \\
       --nodes 1 \\
       --ntasks-per-node 1 \\
       --cpus-per-task 80 \\
       --time 2880 \\
       --output $sweep_folder/$JOB_NAME/stdout.log \\
       --error $sweep_folder/$JOB_NAME/stderr.log \\
       $train_script
"""
)

# note all the models will be in folder ./models/
train_template = string.Template(
"""\
#!/bin/bash
python $ROOT/main.py \\
       --game $GAME \\
       --save_dir $SAVE_DIR \\
       --num_thread 80 \\
       --num_game_per_thread 20 \\
       --actor_sync_freq $ACTOR_SYNC_FREQ \\
       --epoch_len 200 \\
       --batchsize 64 \\
       --burn_in_frames 4000 \\
       --replay_buffer_size 65536 \\
       --priority_exponent 0.9 \\
       --importance_exponent 0.6 \\
       --seq_len $SEQ_LEN \\
       --seq_burn_in $SEQ_BURN_IN \\
       --algo $ALGO \\
       --same_hid $SAME_HID \\
       --one_life 1 \\
       --lr $LR \\
       --eps $EPS \\
"""
)

root = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(root))

default_args = {
    'ROOT': root,
    'JOB_NAME': 'default',
    'GAME': 'pong',
    'SEQ_BURN_IN': 40,
    'SEQ_LEN': 80,
    'ALGO': 'r2d2',
    'ACTOR_SYNC_FREQ': 10,
    'SAME_HID': 0,
    'LR': 1e-4,
    'EPS': 1e-3,
}

games = [
    "asterix",
    "asteroids",
    "bowling",
    "gravitar",
    "ms_pacman",
    # "private_eye",
    "qbert",
    "seaquest",
]

# sweep_folder = "sweep/fake_r2d2_debug"
# variables = OrderedDict([
#     ("GAME", games),
#     ("SEQ_BURN_IN", [0]),
#     ("SEQ_LEN", [1]),
#     # ("ACTOR_SYNC_FREQ", [5, 10]),
#     ("ALGO", ["fake_r2d2"])
# ])

sweep_folder = "sweep2/r2d2_newnet2"
variables = OrderedDict([
    ("GAME", games),
    ("SAME_HID", [0]),
    ("LR", [1e-4, 1e-3]),
    ("EPS", [1e-7]),
])


sweep_folder = os.path.join(root, sweep_folder)
print('sweep folder:', sweep_folder)

if not os.path.exists(sweep_folder):
    print("make dir:", sweep_folder)
    os.makedirs(sweep_folder)
# copy the the sweep script to the folder
shutil.copy2(os.path.realpath(__file__), sweep_folder)


job_names = ['']
sweep_args = [default_args]
for var in variables:
    new_job_names = []
    new_sweep_args = []
    for val in variables[var]:
        for job_name, args in zip(job_names, sweep_args):
            new_job_name = job_name
            if len(new_job_name) > 0:
                new_job_name += '_'
            new_job_name += '%s%s' % (var, val)
            new_job_names.append(new_job_name)

            new_args = args.copy()
            assert var in new_args, var
            new_args[var] = val
            new_sweep_args.append(new_args)
    job_names = new_job_names
    sweep_args = new_sweep_args


print(job_names)
assert len(job_names) == len(sweep_args)


# # create list of args for sweeping
# job_names = ["game%s" % game for game in variables["game"]]
# sweep_args = [{"root": root, "game": game} for game in variables["game"]]


# generate sweep files (srun, train, eval) for each job
srun_files = []
for job_name, arg in zip(job_names, sweep_args):
    exp_folder = os.path.join(sweep_folder, job_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    # train script
    arg["SAVE_DIR"] = exp_folder
    train = train_template.substitute(arg)
    train_file = os.path.join(sweep_folder, job_name, "train.sh")
    with open(train_file, "w") as f:
        f.write(train)
    # print(train)

    srun_arg = {
        "sweep_folder": sweep_folder,
        "JOB_NAME": job_name,
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
            p = subprocess.Popen(["sh", srun_file], cwd=root)
