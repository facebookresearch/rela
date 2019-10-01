import torch
import torch.nn as nn
from typing import Dict


class ApexAgent(torch.jit.ScriptModule):
    __constants__ = ["multi_step", "gamma"]

    def __init__(self, net_cons, multi_step, gamma):
        super().__init__()
        self.net_cons = net_cons
        self.multi_step = multi_step
        self.gamma = gamma

        self.online_net = net_cons()
        self.target_net = net_cons()

    @classmethod
    def clone(cls, model, **kwargs):
        cloned = cls(model.net_cons, model.multi_step, model.gamma)
        cloned.load_state_dict(model.state_dict())
        return cloned

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
    ) -> torch.Tensor:
        online_q = self.online_net(obs)
        online_qa = online_q.gather(1, action["a"].unsqueeze(1)).squeeze(1)

        online_next_a = self.greedy_act(next_obs)
        bootstrap_q = self.target_net(next_obs)
        bootstrap_qa = bootstrap_q.gather(1, online_next_a.unsqueeze(1)).squeeze(1)
        target = reward + bootstrap * (self.gamma ** self.multi_step) * bootstrap_qa
        return target.detach() - online_qa

    @torch.jit.script_method
    def greedy_act(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        legal_move = obs["legal_move"]
        q = self.online_net(obs).detach()
        print(q)
        legal_q = (1 + q - q.min()) * legal_move
        # legal_q > 0 for legal_move and maintain correct orders
        greedy_action = legal_q.argmax(1)
        return greedy_action

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        greedy_action = self.greedy_act(obs)

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
    ) -> torch.Tensor:
        err = self.td_err(obs, action, reward, bootstrap, next_obs)
        return err.detach().abs().cpu()

    def loss(self, batch):
        """
        returns the loss and priority
        """
        err = self.td_err(
            batch.obs, batch.action, batch.reward, batch.bootstrap, batch.next_obs
        )
        loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        priority = err.detach().abs().cpu()
        return loss, priority
