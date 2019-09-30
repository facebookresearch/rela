#!/bin/bash
python main.py --game berzerk \
       --save_dir r2d2_exps/berzerk \
       --r2d2 \
       --num_thread 80 \
       --num_game_per_thread 20 \
       --num_worker_per_thread 1
