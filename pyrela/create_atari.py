import set_path

set_path.append_sys_path()

import os
import pprint

import torch
import rela
import atari

assert rela.__file__.endswith(".so")
assert atari.__file__.endswith(".so")


def get_rom_path(game):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_path = os.path.join(root, "atari", "roms", "%s.bin" % game)
    if not os.path.exists(rom_path):
        print("Error: cannot find rom at:", rom_path)
        assert False
    return rom_path


def get_num_action(game_name):
    game = create_game(game_name, 1, 1, 1, False, False)
    return game.num_action()


def create_game(
    game_name,
    seed,
    eps,
    max_frame,
    terminal_on_life_loss,
    terminal_signal_on_life_loss,
    *,
    frame_stack=4,
    frame_skip=4,
    no_op_start=30,
    height=84,
    width=84,
):
    rom_path = get_rom_path(game_name)
    game = atari.AtariEnv(
        rom_path,
        eps,
        seed,
        frame_stack,
        frame_skip,
        no_op_start,
        height,
        width,
        max_frame,
        terminal_on_life_loss,
        terminal_signal_on_life_loss,
        # video,
    )
    return game


def create_train_env(
    game_name,
    seed,
    eps,
    max_frame,
    num_thread,
    num_game_per_thread,
    actor_creator,
    game_training_proportion,
    is_heirarchical_multigame,
    *,
    is_apex=True,
    terminal_on_life_loss=False,
    terminal_signal_on_life_loss=True,
):
    if game_training_proportion is not None:
        if sum(game_training_proportion.values()) != 1:
            print("The game proportions must sum to 1")
            assert False
        total_game_number = num_thread * num_game_per_thread
        game_training_proportion = {
            key: value * total_game_number
            for key, value in game_training_proportion.items()
        }
        print("going to create the following games", game_training_proportion)
    else:
        print("not doing heterogeneous dev")
    context = rela.Context()
    games = []
    actors = []
    if game_training_proportion is not None:
        game_name_list = list(game_training_proportion.keys())
        game_name_list.remove(game_name)
        game_name_list = [game_name] + game_name_list
        print("game name list is: ", game_name_list)

    i = 0
    for thread_idx in range(num_thread):
        env = rela.VectorEnv()
        game_index = 0
        thread_game_set = set()
        for game_idx in range(num_game_per_thread):
            if game_training_proportion is not None:
                game_name = game_name_list[i]
                game_index = game_name_list.index(game_name)
                game_training_proportion[game_name] -= 1
                if game_training_proportion[game_name] == 0:
                    i += 1
                thread_game_set.add(game_name)
            game = create_game(
                game_name,
                seed + thread_idx * num_game_per_thread + game_idx,
                eps[thread_idx * num_game_per_thread + game_idx],
                max_frame,
                terminal_on_life_loss,
                terminal_signal_on_life_loss,
            )
            games.append(game)
            env.append(game)
        if is_heirarchical_multigame:
            if len(thread_game_set) != 1:
                assert (False, "on a given thread we have to have the same game")

            # making sure the game index is valid to be used in a heirarchical multigame
            assert game_index is not None and game_index >= 0

        if is_apex:
            actor = actor_creator(thread_idx, game_index)
        else:
            acor = actor_creator(thread_idx)
        thread = rela.BasicThreadLoop(actor, env, False)
        actors.append(actor)
        context.push_env_thread(thread)
    print("Finished creating environments with %d games" % (len(games)))
    return context, games, actors


def create_eval_env(
    game_name,
    num_thread,
    model_locker,
    actor_cls,
    seed,
    max_frame,
    *,
    eval_eps=0,
    terminal_on_life_loss=False,
    terminal_signal_on_life_loss=True,
):
    context = rela.Context()
    games = []
    for i in range(num_thread):
        game = create_game(
            game_name,
            seed + i,
            eval_eps,
            max_frame,
            terminal_on_life_loss,
            terminal_signal_on_life_loss,
        )
        games.append(game)
        env = rela.VectorEnv()
        env.append(game)
        actor = actor_cls(model_locker)
        thread = rela.BasicThreadLoop(actor, env, True)
        context.push_env_thread(thread)
    return context, games
