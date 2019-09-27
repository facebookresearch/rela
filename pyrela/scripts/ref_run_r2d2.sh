# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/bin/bash
python main.py \
       --game asterix \
       --save_dir exps/r2d2/GAMEasterix \
       --num_thread 60 \
       --num_game_per_thread 40 \
       --act_device cuda:1,cuda:2,cuda:3 \
       --actor_sync_freq 10 \
       --epoch_len 200 \
       --batchsize 64 \
       --burn_in_frames 4000 \
       --replay_buffer_size 50000 \
       --priority_exponent 0.9 \
       --importance_exponent 0.6 \
       --seq_len 80 \
       --seq_burn_in 40 \
       --algo r2d2 \
       --same_hid 0 \
       --one_life 1 \
       --lr 0.0001 \
       --eps 0.001 \
       --num_epoch 3000 \
       --prefetch 2 \
