import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Dict
import common_utils


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = ["multi_step", "gamma", "eta", "seq_len", "burn_in"]

    def __init__(self, net_cons, device, multi_step, gamma, eta, seq_len, burn_in, same_hid):
        super().__init__()
        self.net_cons = net_cons
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.same_hid = same_hid

        self.online_net = net_cons(device)
        self.target_net = net_cons(device)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    @classmethod
    def clone(cls, model, device):
        cloned = cls(
            model.net_cons,
            device,
            model.multi_step,
            model.gamma,
            model.eta,
            model.seq_len,
            model.burn_in,
            model.same_hid
        )
        cloned.load_state_dict(model.state_dict())
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def _unsqueeze(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_d = {}
        for k, v in d.items():
            new_d[k] = v.unsqueeze(0)
        return new_d

    @torch.jit.script_method
    def act(
        self, obs: Dict[str, torch.Tensor], hid: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape [batchsize]
        """
        # greedy_action, new_hid = self.greedy_act(obs, hid)
        greedy_action, new_hid = self.online_net.act(obs, hid)

        eps = obs["eps"].squeeze(1)
        random_action = obs["legal_move"].multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(0), device=greedy_action.device)
        rand = (rand < eps).long()
        action = (greedy_action * (1 - rand) + random_action * rand).long()
        return {"a": action.detach().cpu()}, new_hid

    @torch.jit.script_method
    def compute_priority(
        self,
        obs: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        terminal: torch.Tensor,
        bootstrap: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
        next_hid: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        compute priority for one batch
        """
        obs = self._unsqueeze(obs)
        online_q = self.online_net(obs, hid, action["a"].unsqueeze(0))[0].squeeze(0)

        next_action = self.online_net.act(next_obs, next_hid)[0].unsqueeze(0)
        next_obs = self._unsqueeze(next_obs)
        bootstrap_q = self.target_net(next_obs, next_hid, next_action)[0].squeeze(0)
        assert reward.size() == bootstrap.size()
        assert reward.size() == bootstrap_q.size()
        target = reward + bootstrap * (self.gamma ** self.multi_step) * bootstrap_q
        priority = (target - online_q).abs()
        return priority.detach().cpu()

    @torch.jit.script_method
    def aggregate_priority(
        self, priority: torch.Tensor, seq_len: torch.Tensor
    ) -> torch.Tensor:
        """
        Given priority, compute the aggregated priority.
        Assumes priority is float Tensor of size [batchsize, seq_len]
        """
        # print(priority.size(1), self.seq_len, seq_len.max())
        assert priority.size(1) == self.seq_len
        assert priority.sum(1).size() == seq_len.size()
        mask = torch.arange(0, priority.size(1), device=seq_len.device)
        mask = (mask.unsqueeze(0) < seq_len.unsqueeze(1)).float()
        priority = priority * mask

        p_mean = priority.sum(1) / (seq_len - self.burn_in)
        p_max = priority.max(1)[0]
        agg_priority = self.eta * p_max + (1.0 - self.eta) * p_mean
        return agg_priority.detach().cpu()

    def td_err(
        self,
        obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        terminal: torch.Tensor,
        bootstrap: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        assert seq_len.dim() == 1, "1 dim, batchsize"
        terminal = terminal.float()

        burn_in_obs = {}
        train_obs = {}
        for k, v in obs.items():
            burn_in_obs[k] = v[:self.burn_in]
            train_obs[k] = v[self.burn_in:]

        if self.burn_in == 0:
            online_hid = hid
            target_hid = hid
        else:
            with torch.no_grad():
                _, online_hid = self.online_net.unroll_rnn(burn_in_obs, hid)
                if self.same_hid:
                    target_hid = online_hid
                else:
                    _, target_hid = self.target_net.unroll_rnn(burn_in_obs, hid)

                for k in online_hid:
                    # to handle dummy burn_in, at beginning of episode
                    zero_out = (1 - terminal[self.burn_in-1]).unsqueeze(0).unsqueeze(2)
                    # print(k)
                    # print(online_hid[k].size())
                    # print(zero_out.size())
                    online_hid[k] = online_hid[k] * zero_out
                    target_hid[k] = target_hid[k] * zero_out
                    # # TODO::::::
                    # online_hid[k].zero_() # = online_hid[k] * zero_out
                    # target_hid[k].zero_() # = target_hid[k] * zero_out

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        train_action = action["a"][self.burn_in:]
        online_qas, greedy_action = self.online_net(train_obs, online_hid, train_action)
        with torch.no_grad():
            target_qas, _ = self.target_net(train_obs, target_hid, greedy_action)

        assert greedy_action.size() == train_action.size()
        assert online_qas.size() == target_qas.size()

        assert self.seq_len == online_qas.size(0) - self.multi_step
        reward = reward[self.burn_in:]
        terminal = terminal[self.burn_in:]
        bootstrap = bootstrap[self.burn_in:]

        # print('reward:', reward.size())
        # print('terminal:', terminal.size())
        # print('bootstrap:', bootstrap.size())
        # print('online_qas:', online_qas.size())

        errs = []
        for i in range(self.seq_len):
            target_i = i + self.multi_step
            target_qa = target_qas[target_i]
            # if target_i < train_seq_len:
            #     target_qa = target_qas[target_i]
            bootstrap_qa = (self.gamma ** self.multi_step) * target_qa
            target = reward[i] + bootstrap[i] * bootstrap_qa

            # sanity check
            should_padding = i >= (seq_len - self.burn_in)
            if i > 0:
                is_padding = (terminal[i] + terminal[i - 1] == 2).float()
                if not (is_padding.long() == should_padding.long()).all():
                    import pdb
                    pdb.set_trace()
                # assert (is_padding.long() == should_padding.long()).all()

            err = (target.detach() - online_qas[i]) * (1 - should_padding.float())
            errs.append(err)

        return torch.stack(errs, 1)

    def loss(self, batch):
        err = self.td_err(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len,
        )
        loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        # sum over seq dim
        # err is [batch, seq_len], required by aggregate_priority
        loss = loss.sum(1)
        priority = self.aggregate_priority(err.abs(), batch.seq_len)
        return loss, priority
