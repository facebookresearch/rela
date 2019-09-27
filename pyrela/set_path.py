# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def append_sys_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(root)
    rela = os.path.join(root, "atari", "build", "rela")
    if rela not in sys.path:
        sys.path.append(rela)
    atari = os.path.join(root, "atari", "build")
    if atari not in sys.path:
        sys.path.append(atari)

    # # TODO: move python to another folder
    # pyrela = os.path.join(root, "rela")
    # if pyrela not in sys.path:
    #     sys.path.append(pyrela)
