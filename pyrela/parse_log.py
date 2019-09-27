import os
from collections import defaultdict
import numpy as np


def shorten_name(s):
    s = s.replace("NUM_EPOCH20_", "")
    s = s.replace("GAME_PER_THREAD", "GPT")
    s = s.replace("cuda:", "")
    s = s.replace("GAME", "")
    s = s.replace("NUM_", "")
    s = s.replace("THREAD", "T")
    return s


def timestr_to_min(timestr):
    timestr = timestr.strip()
    h, m, s = timestr.split()
    h = float(h[:-1])
    m = float(m[:-1])
    m = h * 60 + m
    return m


def timestr_to_hour(timestr):
    timestr = timestr.strip()
    h, m, s = timestr.split()
    h = float(h[:-1])
    m = float(m[:-1])
    m = h + m / 60
    return m


def timestr_to_hour(timestr):
    timestr = timestr.strip()
    h, m, s = timestr.split()
    h = float(h[:-1])
    m = float(m[:-1])
    m = h + m / 60
    return m


def numstr_to_num(numstr):
    numstr = numstr.strip()
    num = float(numstr[:-1])
    unit = numstr[-1]
    if unit == "M":
        return num * 1e6
    if unit == "K":
        return num * 1e3


def parse_log(filename, max_epoch):
    lines = open(filename, "r").readlines()
    scores = []
    # perfects = []
    train_rates = []
    buffer_rates = []
    act_rates = []
    # pred_loss = []
    calc_loss = []
    back_prop = []
    sample_update = []
    times = []
    samples = []

    for l in lines:
        if "Speed" in l:
            rates = l.split()
            train_rate = float(rates[2][:-1])
            act_rate = float(rates[4][:-1])
            buffer_rate = float(rates[6][:-1])
            train_rates.append(train_rate)
            act_rates.append(act_rate)
            buffer_rates.append(buffer_rate)
        if "calculating loss" in l:
            t = float(l.split()[3])
            calc_loss.append(t)
        if "backprop & update" in l:
            t = float(l.split()[3])
            back_prop.append(t)
        if "sample data" in l:
            t = float(l.split()[3])
            sample_update.append(t)
        if "updating priority" in l:
            t = float(l.split()[2])
            sample_update[-1] += t
        if "eval score:" in l:
            score = float(l.split()[-1])
            scores.append(score)
        if "total time(train)" in l:
            t = l.split(":")[-1]
            times.append(timestr_to_hour(t))
        if "Total Sample" in l:
            s = numstr_to_num(l.split(" ")[-3][:-1])
            samples.append(s)
        if max_epoch > 0 and len(train_rates) == max_epoch:
            break

    epoch = len(train_rates)
    avg_act_rate = int(np.mean(act_rates[-10:]))
    avg_train_rate = int(np.mean(train_rates[-10:]))
    avg_buffer_rate = int(np.mean(buffer_rates[-10:]))
    return {
        "id": filename,
        "epoch": epoch,
        "act_rate": avg_act_rate,
        "train_rate": avg_train_rate,
        "buffer_rate": avg_buffer_rate,
        "calc_loss": np.mean(calc_loss[-10:]),
        "back_prop": np.mean(back_prop[-10:]),
        "sample": np.mean(sample_update[-10:]),
        "final_score": np.mean(scores[-10:]),
        "scores": scores,
        "times": times[: len(scores)],
        "samples": samples[: len(scores)],
    }


def average_across_seed(logs):
    new_logs = defaultdict(list)
    for k, v in logs.items():
        s = k.rsplit("_", 1)
        if len(s) == 2:
            name, seed = s
        elif len(s) == 1:
            name = "default"
            seed = s[0]
        if not seed.startswith("SEED"):
            print("no multiple seeds, omit averaging: ", name)
            name = k
        new_logs[name].append(v)

    for k in new_logs:
        vals = new_logs[k]
        min_len = np.min([len(v) for v in vals])
        assert min_len > 0, min_len
        vals = np.stack([v[:min_len] for v in vals])
        # print(k, vals.shape)
        # new_vals = [np.mean([v[i] for v in vals]) for i in range(len(vals[0]))]
        mean = vals.mean(0).tolist()
        sem = vals.std(0) / np.sqrt(vals.shape[0])
        new_logs[k] = (mean, sem)

    # l = list(new_logs.items())
    # l = sorted(l, key=lambda x: -x[1][-2])
    # summary = [(shorten_name(ll[0]), *ll[1]) for ll in l]
    # header = ['name', 'epoch', 'act', 'train', 'buffer', 'score', 'perfect']
    # print(tabulate(summary, headers=header))
    return new_logs


def max_across_seed(logs):
    new_logs = {}
    for k, v in logs.items():
        s = k.rsplit("_", 1)
        if len(s) == 2:
            name, seed = s
        elif len(s) == 1:
            name = "default"
            seed = s[0]
        if not seed.startswith("SEED"):
            print("no multiple seeds, omit averaging: ", name)
            name = k
        if name not in new_logs or np.mean(v[-10:]) > new_logs[name][0]:
            new_logs[name] = (np.mean(v[-10:]), k)

    return new_logs


def parse_from_root(root, max_epoch, min_epoch, filter_string):
    logs = {}
    root = os.path.abspath(root)
    for exp in os.listdir(root):
        if filter_string:
            go = True
            for s in filter_string:
                if s not in exp:
                    go = False
                    break
            if not go:
                continue
        exp_folder = os.path.join(root, exp)
        log_file = os.path.join(exp_folder, "train.log")
        if os.path.exists(log_file):
            try:
                log = parse_log(log_file, max_epoch)
                if min_epoch > 0 and log["epoch"] < min_epoch:
                    print(
                        "%s is dropped due to being too short\n\t%d vs %d"
                        % (log_file, log["epoch"], min_epoch)
                    )
                else:
                    logs[exp] = log
            except:
                print("something is wrong with %s" % log_file)

    return logs
