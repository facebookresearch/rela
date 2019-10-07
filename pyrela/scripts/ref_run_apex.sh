export job_name='exps/apex_pong_no_prefetch'
mkdir -p ${job_name}
sbatch --job-name $job_name \
       --mem 500G \
       --partition dev \
       --constraint pascal \
       --gres gpu:2 \
       --nodes 1 \
       --ntasks-per-node 1 \
       --cpus-per-task 80 \
       --output ${job_name}/std.out \
       --error ${job_name}/std.err \
       --time 2880 \
       --wrap "
#!/bin/bash
python main.py \
       --game pong \
       --save_dir ${job_name} \
       --num_thread 80 \
       --num_game_per_thread 20 \
       --lr 6.25e-5 \
       --eps 1.5e-7 \
       --algo apex \
       --actor_sync_freq 20 \
"
