# !/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

python main.py --save_dir exps/dev \
       --num_thread 1 \
       --num_game_per_thread 1 \
       --epoch_len 200 \
       --game boxing  \
       --algo r2d2 \
       --burn_in_frames 50 \
       --batchsize 32 \


# python main.py \
#        --game pong \
#        --save_dir exps/dev \
#        --num_thread 80 \
#        --num_game_per_thread 20 \
#        --lr 6.25e-5 \
#        --eps 1.5e-7 \
#        --algo apex \
#        --actor_sync_freq 20 \
#        # --burn_in_frames 500 \
