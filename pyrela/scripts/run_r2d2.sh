export game=boxing
export suffix=stored_priority
export job_name=exps/r2d2_${game}_${suffix}
mkdir -p ${job_name}
sbatch --job-name $job_name \
       --exclusive \
       --mem 500G \
       --partition learnfair \
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
python main.py --game ${game} \
       --save_dir ${job_name} \
       --algo r2d2 \
       --num_thread 80 \
       --num_game_per_thread 20 \
       --actor_sync_freq 10 \
       --epoch_len 200 \
       --batchsize 64 \
       --burn_in_frames 4000 \
       --replay_buffer_size 65536 \
       --priority_exponent 0.9 \
       --importance_exponent 0.6 \
       --lr 1e-4 \
       --eps 1e-3 \
"
