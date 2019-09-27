# !/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

python main.py \
       --save_dir exps/dev \
       --num_thread 1 \
       --num_game_per_thread 1 \
       --epoch_len 200 \
       --game boxing  \
       --algo apex \
       --burn_in_frames 50 \
       --batchsize 4 \
