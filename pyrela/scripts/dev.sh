#!/bin/bash
python main.py --save_dir exps/dev \
       --num_thread 1 \
       --num_game_per_thread 1 \
       --epoch_len 10 \
       --game boxing  \
       --algo apex \
       --burn_in_frames 500 \
       --batchsize 32 \
