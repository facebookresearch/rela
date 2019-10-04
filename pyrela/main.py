import time
import os
import sys
import argparse
import pprint

import numpy as np
import torch

import create_atari
import rela
from apex import ApexAgent
from r2d2 import R2D2Agent
from net import AtariFFNet, AtariLSTMNet

import common_utils
import utils
from eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Run DQN on Atari")
    parser.add_argument("--save_dir", default="exps/exp1", type=str)
    parser.add_argument("--grad_clip", default=5, type=float)

    # R2D2 settings
    parser.add_argument("--algo", type=str, required=True, help="apex/r2d2")
    parser.add_argument("--seq_burn_in", type=int, default=40)
    parser.add_argument("--seq_len", type=int, default=80)
    parser.add_argument("--eta", type=float, default=0.9)

    # game settings
    parser.add_argument("--game", type=str, default="boxing")
    parser.add_argument("--seed", type=int, default=10002)
    parser.add_argument(
        "--max_frame", type=int, default=108000, help="30min of gameplay (50fps)"
    )
    parser.add_argument("--gamma", type=float, default=0.997, help="discount factor")

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="optim epsilon")
    # parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    # parser.add_argument("--adam_eps", type=float, default=1e-3, help="Adam epsilon")
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", default=512, type=int, help="for train")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    # DQN settings
    parser.add_argument("--hid_dim", type=int, default=512, help="size of hid layer")
    parser.add_argument("--multi_step", type=int, default=3, help="multi-step return")

    # replay buffer settings
    parser.add_argument("--burn_in_frames", default=80000, type=int)
    parser.add_argument("--replay_buffer_size", default=2 ** 21, type=int)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.6, help="alpha in PER paper"
    )
    parser.add_argument(
        "--importance_exponent", type=float, default=0.4, help="beta in PER paper"
    )

    # thread setting
    parser.add_argument("--num_thread", default=40, type=int)
    parser.add_argument("--num_game_per_thread", default=20, type=int)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.4)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", default="cuda:1", type=str)
    parser.add_argument("--actor_sync_freq", default=20, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benckmark = True
    args = parse_args()

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    sys.stdout = common_utils.Logger(os.path.join(args.save_dir, "train.log"))
    sys.stderr = common_utils.Logger(os.path.join(args.save_dir, "train.err"))

    num_action = create_atari.get_num_action(args.game)
    if args.algo == "r2d2":
        net_cons = lambda device: AtariLSTMNet(device, num_action)
        agent = R2D2Agent(
            net_cons, args.train_device, args.multi_step, args.gamma, args.eta, args.seq_burn_in
        )
        replay_class = rela.RNNPrioritizedReplay
    elif args.algo == "apex":
        net_cons = lambda: AtariFFNet(num_action)
        agent = ApexAgent(net_cons, args.multi_step, args.gamma)
        replay_class = rela.FFPrioritizedReplay

    agent = agent.to(args.train_device)
    print(agent)

    if args.algo == "apex":
        optim = torch.optim.RMSprop(
            agent.online_net.parameters(), lr=args.lr, eps=args.eps
        )
    elif args.algo == "r2d2":
        optim = torch.optim.Adam(
            agent.online_net.parameters(), lr=args.lr, eps=args.eps
        )

    # to keep them alive
    ref_models = [agent.clone(agent, args.act_device) for _ in range(3)]
    model_locker = rela.ModelLocker(ref_models, args.act_device)
    replay_buffer = replay_class(
        args.replay_buffer_size,
        args.seed,
        args.priority_exponent,
        args.importance_exponent,
        args.batchsize,
    )

    explore_eps = utils.generate_eps(
        args.act_base_eps,
        args.act_eps_alpha,
        args.num_thread * args.num_game_per_thread,
    )

    if args.algo == "r2d2":
        actor_cls = rela.R2D2Actor
        actor_creator = lambda i: rela.R2D2Actor(
            model_locker,
            args.multi_step,
            args.num_game_per_thread,
            args.gamma,
            args.seq_len,
            args.seq_burn_in,
            replay_buffer,
        )
    elif args.algo == "apex":
        actor_cls = rela.DQNActor
        actor_creator = lambda i: rela.DQNActor(
            model_locker,
            args.multi_step,
            args.num_game_per_thread,
            args.gamma,
            replay_buffer,
        )

    print("creating train env")
    context, games, actors = create_atari.create_train_env(
        args.game,
        args.seed,
        explore_eps,
        args.max_frame,
        args.num_thread,
        args.num_game_per_thread,
        actor_creator,
    )

    context.start()
    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)

    stopwatch = common_utils.Stopwatch()
    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    train_time = 0
    for epoch in range(args.num_epoch):
        mem_usage = common_utils.get_mem_usage()
        print("Beginning of Epoch %d\nMem usage: %s" % (epoch, mem_usage))

        stat.reset()
        stopwatch.reset()
        tachometer.start()
        t = time.time()
        for batch_idx in range(args.epoch_len):
            num_update = batch_idx + epoch * args.epoch_len
            if stopwatch is not None:
                torch.cuda.synchronize()
            if num_update % args.num_update_between_sync == 0:
                agent.sync_target_with_online()

            if num_update % args.actor_sync_freq == 0:
                # ref_model = agent.clone(agent, device=args.act_device)._c
                model_locker.update_model(agent)
                # model_locker.update_model(ref_model)
            if stopwatch is not None:
                torch.cuda.synchronize()
                stopwatch.time("sync and updating")

            batch, weight = replay_buffer.sample()
            stopwatch.time("sample data")

            if stopwatch is not None:
                torch.cuda.synchronize()
            weight = weight.to(args.train_device).detach()
            batch = utils.to_device(batch, args.train_device)
            if stopwatch is not None:
                torch.cuda.synchronize()
                stopwatch.time("device conversion")

            if stopwatch is not None:
                torch.cuda.synchronize()

            loss, priority = agent.loss(batch)
            loss = (loss * weight).mean()
            if stopwatch is not None:
                torch.cuda.synchronize()
                stopwatch.time("calculating loss")

            if stopwatch is not None:
                torch.cuda.synchronize()
            loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(
                agent.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()
            if stopwatch is not None:
                torch.cuda.synchronize()
                stopwatch.time("backprop & update")

            replay_buffer.update_priority(priority)
            if stopwatch is not None:
                stopwatch.time("updating priority")

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)

        epoch_t = time.time() - t
        train_time += epoch_t
        print(
            "epoch: %d, time: %.1fs, total time(train): %s"
            % (epoch, epoch_t, common_utils.sec2str(train_time))
        )
        tachometer.lap(actors, replay_buffer, args.epoch_len * args.batchsize)
        if stopwatch is not None:
            stopwatch.summary()

        context.pause()
        eval_model = agent.clone(agent, device="cpu")
        score = evaluate(
            args.game,
            10,
            eval_model,
            "cpu",
            actor_cls,
            epoch + 999,
            args.max_frame,
            0,
        )

        print("epoch %d, eval score: %f" % (epoch, score))
        stat["eval_score"].feed(score)
        context.resume()

        stat.summary(epoch)
        print("****************************************")
    replay_buffer.signal_done()
