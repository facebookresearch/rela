import torch
import torch.nn as nn
from typing import Dict


class ApexAgent(torch.jit.ScriptModule):
    __constants__ = ["multi_step", "gamma", "use_heirarchical_net", "total_num_games"]

    def __init__(
        self, net_cons, multi_step, gamma, use_heirarchical_net, total_num_games
    ):
        super().__init__()
        self.net_cons = net_cons
        self.multi_step = multi_step
        self.gamma = gamma
        self.use_heirarchical_net = use_heirarchical_net
        self.total_num_games = total_num_games

        self.online_net = net_cons()
        self.target_net = net_cons()

    @classmethod
    def clone(cls, model, device):
        cloned = cls(
            model.net_cons,
            model.multi_step,
            model.gamma,
            model.use_heirarchical_net,
            model.total_num_games,
        )
        cloned.load_state_dict(model.state_dict())
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def td_err(
        self,
        obs: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        bootstrap: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        game_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_heirarchical_net:
            target = torch.zeros(bootstrap.shape, device=bootstrap.device)
            online_qa = torch.zeros(bootstrap.shape, device=bootstrap.device)
            for i in range(self.total_num_games):
                curr_indices = (obs["game_idx"] == i).nonzero()
                if len(curr_indices.shape) == 2:
                    if curr_indices.shape == (1, 1):
                        curr_indices = torch.squeeze(curr_indices, dim=1)
                    else:
                        curr_indices = torch.squeeze(curr_indices)
                if len(curr_indices.size()) == 0 or len(curr_indices) == 0:
                    continue

                (
                    curr_obs,
                    curr_action,
                    curr_reward,
                    curr_bootstrap,
                    curr_next_obs,
                    curr_game_idx,
                ) = self.batch_inputs_on_game_idx(
                    obs, action, reward, bootstrap, next_obs, game_idx, curr_indices
                )

                curr_online_q = self.online_net(curr_obs)
                curr_online_qa = curr_online_q.gather(
                    1, curr_action["a"].unsqueeze(1)
                ).squeeze(1)

                curr_online_next_a = self.greedy_act(curr_next_obs, curr_game_idx)
                curr_bootstrap_q = self.target_net(curr_next_obs)
                curr_bootstrap_qa = curr_bootstrap_q.gather(
                    1, curr_online_next_a.unsqueeze(1)
                ).squeeze(1)

                target[curr_indices] = (
                    curr_reward
                    + curr_bootstrap
                    * (self.gamma ** self.multi_step)
                    * curr_bootstrap_qa
                )
                online_qa[curr_indices] = curr_online_qa

            return target.detach() - online_qa

        else:
            online_q = self.online_net(obs)
            online_qa = online_q.gather(1, action["a"].unsqueeze(1)).squeeze(1)

            online_next_a = self.greedy_act(next_obs, game_idx)
            bootstrap_q = self.target_net(next_obs)
            bootstrap_qa = bootstrap_q.gather(1, online_next_a.unsqueeze(1)).squeeze(1)
            target = reward + bootstrap * (self.gamma ** self.multi_step) * bootstrap_qa

        return target.detach() - online_qa

    @torch.jit.script_method
    def greedy_act(
        self, obs: Dict[str, torch.Tensor], game_idx: torch.Tensor
    ) -> torch.Tensor:
        legal_move = obs["legal_move"]
        game_idx = obs["game_idx"]
        if self.use_heirarchical_net:
            q = torch.zeros(legal_move.shape, device=obs["s"].device)

            for i in range(self.total_num_games):
                curr_indices = (obs["game_idx"] == i).nonzero()
                if len(curr_indices.shape) == 2:
                    if curr_indices.shape == (1, 1):
                        curr_indices = torch.squeeze(curr_indices, dim=1)
                    else:
                        curr_indices = torch.squeeze(curr_indices)
                if len(curr_indices.size()) == 0 or len(curr_indices) == 0:
                    continue
                curr_obs = self.batch_obs_on_game_idx(obs, curr_indices)
                q[curr_indices] = self.online_net(curr_obs).detach()
        else:
            q = self.online_net(obs).detach()
        legal_q = (1 + q - q.min()) * legal_move
        # legal_q > 0 for legal_move and maintain correct orders
        greedy_action = legal_q.argmax(1)
        return greedy_action

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        game_idx = obs["game_idx"]

        greedy_action = self.greedy_act(obs, game_idx)

        eps = obs["eps"].squeeze(1)
        random_action = obs["legal_move"].multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(0), device=greedy_action.device)
        rand = (rand < eps).long()
        action = (greedy_action * (1 - rand) + random_action * rand).long()
        return {"a": action.detach().cpu()}

    @torch.jit.script_method
    def compute_priority(
        self,
        obs: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        terminal: torch.Tensor,
        bootstrap: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        game_idx: torch.Tensor,
    ) -> torch.Tensor:
        err = self.td_err(obs, action, reward, bootstrap, next_obs, game_idx)
        return err.detach().abs().cpu()

    def loss(self, batch):
        """
        returns the loss and priority
        """
        err = self.td_err(
            batch.obs,
            batch.action,
            batch.reward,
            batch.bootstrap,
            batch.next_obs,
            batch.game_idx,
        )
        loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        priority = err.detach().abs().cpu()
        return loss, priority

    @torch.jit.script_method
    def batch_inputs_on_game_idx(
        self,
        obs: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        bootstrap: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        game_idx: torch.Tensor,
        indices: torch.Tensor,
    ):

        new_obs = self.batch_dict_on_indices(obs, indices)
        new_action = self.batch_dict_on_indices(action, indices)
        new_reward = self.batch_tensor_on_indices(reward, indices)
        new_bootstrap = self.batch_tensor_on_indices(bootstrap, indices)
        new_next_obs = self.batch_dict_on_indices(next_obs, indices)
        new_game_idx = self.batch_tensor_on_indices(game_idx, indices)
        return (
            new_obs,
            new_action,
            new_reward,
            new_bootstrap,
            new_next_obs,
            new_game_idx,
        )

    @torch.jit.script_method
    def batch_obs_on_game_idx(
        self, obs: Dict[str, torch.Tensor], indices: torch.Tensor
    ):
        return self.batch_dict_on_indices(obs, indices)

    @torch.jit.script_method
    def batch_dict_on_indices(
        self, input_dict: Dict[str, torch.Tensor], indices: torch.Tensor
    ):
        output_dict = {}
        for key in input_dict.keys():
            output_dict[key] = input_dict[key][indices]
            continue
        return output_dict

    @torch.jit.script_method
    def batch_tensor_on_indices(
        self, input_tensor: torch.Tensor, indices: torch.Tensor
    ):
        return input_tensor[indices]
