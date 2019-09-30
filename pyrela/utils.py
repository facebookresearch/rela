import set_path

set_path.append_sys_path()

import time
import torch
import rela
import common_utils


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device).detach()
    elif isinstance(batch, dict):
        return {key: to_device(batch[key], device) for key in batch}
    elif isinstance(batch, rela.FFTransition):
        batch.obs = to_device(batch.obs, device)
        batch.action = to_device(batch.action, device)
        batch.reward = to_device(batch.reward, device)
        batch.terminal = to_device(batch.terminal, device)
        batch.bootstrap = to_device(batch.bootstrap, device)
        batch.next_obs = to_device(batch.next_obs, device)
        return batch
    elif isinstance(batch, rela.RNNTransition):
        batch.obs = to_device(batch.obs, device)
        batch.h0 = to_device(batch.h0, device)
        batch.action = to_device(batch.action, device)
        batch.reward = to_device(batch.reward, device)
        batch.terminal = to_device(batch.terminal, device)
        batch.bootstrap = to_device(batch.bootstrap, device)
        batch.seq_len = to_device(batch.seq_len, device)
        return batch
    else:
        assert False, "unsupported type: %s" % type(batch)


# returns the number of steps in all actors
def get_num_acts(actors):
    total_acts = 0
    for actor in actors:
        total_acts += actor.num_act()
    return total_acts


class Tachometer:
    def __init__(self):
        self.num_act = 0
        self.num_buffer = 0
        self.num_train = 0
        self.t = None

    def start(self):
        self.t = time.time()

    def lap(self, actors, replay_buffer, num_train):
        t = time.time() - self.t
        num_act = get_num_acts(actors)
        act_rate = (num_act - self.num_act) / t
        num_buffer = replay_buffer.num_add()
        buffer_rate = num_buffer - self.num_buffer
        train_rate = num_train / t
        print(
            "Speed: train: %.1f, act: %.1f, buffer-add: %.1f"
            % (train_rate, act_rate, buffer_rate)
        )
        self.num_act = num_act
        self.num_buffer = num_buffer
        self.num_train += num_train
        print(
            "Total Sample: train: %s, act: %s"
            % (common_utils.num2str(self.num_train), common_utils.num2str(self.num_act))
        )


# num_acts is the total number of acts, so total number of acts is num_acts * num_game_per_actor
# num_buffer is the total number of elements inserted into the buffer
# time elapsed is in seconds
def get_frame_stat(num_game_per_thread, time_elapsed, num_acts, num_buffer, frame_stat):
    act_rate = (num_acts - frame_stat["num_acts"]) / time_elapsed
    buffer_rate = (num_buffer - frame_stat["num_buffer"]) / time_elapsed
    frame_stat["num_acts"] = num_acts
    frame_stat["num_buffer"] = num_buffer
    return (act_rate, buffer_rate)


def generate_eps(base_eps, alpha, num_actor):
    if num_actor == 1:
        return [base_eps]

    eps_list = []
    for i in range(num_actor):
        eps = base_eps ** (1 + i / (num_actor - 1) * alpha)
        eps_list.append(eps)
    return eps_list
