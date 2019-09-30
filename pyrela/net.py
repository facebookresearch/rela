import torch
import torch.nn as nn
from typing import Dict


class AtariFFNet(torch.jit.ScriptModule):
    __constants__ = ["conv_out", "num_action"]

    def __init__(self, num_action):
        super().__init__()
        self.frame_stack = 4
        self.conv_out = 3136
        self.self.hid_dim = 512
        self.num_action = num_action

        self.net = nn.Sequential(
            nn.Conv2d(self.frame_stack, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(nn.Linear(self.conv_out, self.hid_dim), nn.ReLU())
        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.num_action)

    @torch.jit.script_method
    def duel(
        self, v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor
    ) -> torch.Tensor:
        assert a.size() == legal_move.size()
        legal_a = a * legal_move
        q = v + legal_a - legal_a.mean(1, keepdim=True)
        return q

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        q-values for invalid_moves are UNDEFINED
        """
        s = obs["s"].float() / 255.0
        legal_move = obs["legal_move"]

        h = self.net(s).view(-1, self.conv_out)
        h = self.linear(h)
        v = self.fc_v(h)  # .view(-1, 1)
        a = self.fc_a(h)  # .view(-1, self.num_action)

        return self.duel(v, a, legal_move)
