#!/bin/bash
python /private/home/hengyuan/rela/pyrela/main.py \
       --game asterix \
       --save_dir /private/home/hengyuan/rela/pyrela/sweep2/apex_newnet/GAMEasterix_MULTI_STEP3 \
       --num_thread 80 \
       --num_game_per_thread 20 \
       --lr 6.25e-5 \
       --eps 1.5e-7 \
       --algo apex \
       --actor_sync_freq 20 \
       --one_life 1 \
       --multi_step 3 \
