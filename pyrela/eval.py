import time
import numpy as np
from create_atari import create_eval_env, get_num_action
import rela

# import utils


def evaluate(
    game_name, num_thread, model, device, actor_cls, seed, max_frame, eval_eps
):
    model_locker = rela.ModelLocker(model._c, device)
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
    from net import AtariFFNet
    from apex import ApexAgent

    # from r2d2 import R2D2Agent

    save_dir = "exps/test_eval"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    game = "pong"
    device = "cpu"
    num_action = get_num_action(game)
    print("game: %s, num action: %d" % (game, num_action))

    net_cons = lambda: AtariFFNet(num_action)
    model = ApexAgent(net_cons, 1, 0.99)
    actor_cls = rela.DQNActor

    score = evaluate(game, 10, model, device, actor_cls, 999, 108000, 0)
    print("score: ", score)
