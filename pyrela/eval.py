# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time
import numpy as np
from create_atari import create_eval_env, get_num_action
import rela


def evaluate(
    game_name,
    num_thread,
    model_locker,
    actor_cls,
    seed,
    max_frame,
    eval_eps,
    terminal_on_life_loss,
):
    context, games = create_eval_env(
        game_name,
        num_thread,
        model_locker,
        actor_cls,
        seed,
        max_frame,
        eval_eps=eval_eps,
    )
    context.start()
    while not context.terminated():
        time.sleep(0.5)

    context.terminate()
    while not context.terminated():
        time.sleep(0.5)
    scores = [g.get_episode_reward() for g in games]
    return np.mean(scores)


if __name__ == "__main__":
    import os
    import torch
    from net import AtariFFNet, AtariLSTMNet
    from apex import ApexAgent
    from r2d2 import R2D2Agent

    save_dir = "exps/test_eval"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    game = "pong"
    device = "cpu"
    num_action = get_num_action(game)
    print("game: %s, num action: %d" % (game, num_action))

    # net_cons = lambda: AtariFFNet(num_action)
    # model = ApexAgent(net_cons, 1, 0.99)
    # actor_cls = rela.DQNActor

    net_cons = lambda: AtariLSTMNet(device, num_action)
    model = R2D2Agent(net_cons, 1, 0.99, 0.9)
    locker = rela.ModelLocker([model], device)
    actor_cls = rela.R2D2Actor

    score = evaluate(game, 10, locker, actor_cls, 999, 108000, 0)
    print("score: ", score)
