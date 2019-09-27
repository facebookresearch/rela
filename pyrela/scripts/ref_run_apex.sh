# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/bin/bash
python main.py \
       --game asterix \
       --save_dir exps/apex/GAMEasterix \
       --num_thread 60 \
       --num_game_per_thread 40 \
       --act_device cuda:1,cuda:2,cuda:3 \
       --lr 6.25e-5 \
       --eps 1.5e-7 \
       --num_epoch 4000 \
       --epoch_len 2000 \
       --algo apex \
       --actor_sync_freq 20 \
       --one_life 1 \
       --multi_step 3 \
       --prefetch 3 \
