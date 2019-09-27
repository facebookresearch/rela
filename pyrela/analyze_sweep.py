import os
import argparse
import pprint
import numpy as np
from tabulate import tabulate

from parse_log import *


# def print_logs(logs):
#     l = list(logs.items())
#     l = sorted(l, key=lambda x: x[0])
#     summary = [
#         (
#             shorten_name(ll[0]),
#             ll[1]["epoch"],
#             ll[1]["act_rate"],
#             ll[1]["train_rate"],
#             ll[1]["buffer_rate"],
#             ll[1]["calc_loss"],
#             ll[1]["back_prop"],
#         )
#         for ll in l
#     ]
#     header = ["name", "epoch", "act", "train", "buffer"]
#     print(tabulate(summary, headers=header))


def analyze_sweep(root, max_epoch, min_epoch, filter_string):
    logs = parse_from_root(root, max_epoch, min_epoch, filter_string)

    l = list(logs.items())
    l = sorted(l, key=lambda x: shorten_name(x[0]))
    summary = [
        (
            shorten_name(ll[0]),
            ll[1]["epoch"],
            ll[1]["act_rate"],
            ll[1]["train_rate"],
            ll[1]["buffer_rate"],
            ll[1]["final_score"],
            ll[1]["calc_loss"],
            ll[1]["back_prop"],
            ll[1]["sample"],
        )
        for ll in l
    ]
    # header = ["name", "epoch", "act", "train", "buffer", "score"]
    header = [
        "name",
        "epoch",
        "act",
        "train",
        "buffer",
        "score",
        "calc_l",
        "bp",
        "sample",
    ]
    print(tabulate(summary, headers=header))

    # print('\n=====avg, stderr======')
    # avg_scores = {k: v['scores'] for k, v in logs.items()}
    # avg_scores = average_across_seed(avg_scores)

    # l = list(avg_scores.items())
    # l = sorted(l, key=lambda x: x[0])
    # summary = [(
    #     shorten_name(ll[0]),
    #     len(ll[1][0]),
    #     np.mean(ll[1][0][-10:]),
    # ) for ll in l]
    # header = ['name', 'epoch', 'score']
    # print(tabulate(summary, headers=header))

    # print('\n======best over seed======')
    # scores = {k: v['scores'] for k, v in logs.items()}
    # best_scores = max_across_seed(scores)
    # for k, (s, loc) in best_scores.items():
    #     print('%s: %.2f' % (k, s))
    #     print('\tat: ', loc)
    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--max_epoch", type=int, default=0)
    parser.add_argument("--min_epoch", type=int, default=0)
    parser.add_argument("--filter_string", type=str, default="", nargs="+")
    args = parser.parse_args()

    analyze_sweep(args.root, args.max_epoch, args.min_epoch, args.filter_string)
