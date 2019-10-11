#!/bin/bash
python /private/home/hengyuan/rela/pyrela/main.py \
       --game bowling \
       --save_dir /private/home/hengyuan/rela/pyrela/sweep2/r2d2_onelife_net/GAMEbowling_SAME_HID1 \
       --num_thread 80 \
       --num_game_per_thread 20 \
       --actor_sync_freq 10 \
       --epoch_len 200 \
       --batchsize 64 \
       --burn_in_frames 4000 \
       --replay_buffer_size 65536 \
       --priority_exponent 0.9 \
       --importance_exponent 0.6 \
       --seq_len 80 \
       --seq_burn_in 40 \
       --algo r2d2 \
       --same_hid 1 \
       --one_life 1 \
       --lr 0.0001 \
