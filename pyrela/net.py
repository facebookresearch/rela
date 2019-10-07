import torch
import torch.nn as nn
from typing import Dict, Tuple


class AtariFFNet(torch.jit.ScriptModule):
    __constants__ = ["conv_out_dim", "num_action"]

    def __init__(self, num_action):
        super().__init__()
        self.frame_stack = 4
        self.conv_out_dim = 3136
        self.hid_dim = 512
        self.num_action = num_action

        self.net = nn.Sequential(
            nn.Conv2d(self.frame_stack, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(self.conv_out_dim, self.hid_dim), nn.ReLU()
        )
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
        h = self.net(s).view(-1, self.conv_out_dim)
        h = self.linear(h)
        v = self.fc_v(h)  # .view(-1, 1)
        a = self.fc_a(h)  # .view(-1, self.num_action)
        
        return self.duel(v, a, legal_move)


class AtariLSTMNet(torch.jit.ScriptModule):
    __constants__ = ["conv_out_dim", "hid_dim", "num_lstm_layer"]

    def __init__(self, device, num_action):
        super().__init__()
        self.frame_stack = 4
        self.conv_out_dim = 3136
        self.hid_dim = 512
        self.num_lstm_layer = 1
        self.num_action = num_action

        self.net = nn.Sequential(
            nn.Conv2d(self.frame_stack, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(self.conv_out_dim, self.hid_dim), nn.ReLU()
        )
        self.lstm = nn.LSTM(
            self.hid_dim, self.hid_dim, num_layers=self.num_lstm_layer
        ).to(device)
        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.num_action)

        self.lstm.flatten_parameters()

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)

        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def duel(
        self, v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor
    ) -> torch.Tensor:
        assert a.size() == legal_move.size()
        legal_a = a * legal_move
        q = v + legal_a - legal_a.mean(2, keepdim=True)
        return q

    @torch.jit.script_method
    def _conv_forward(self, s: torch.Tensor):
        assert s.dim() == 4  # [batch, c, h, w]
        s = s.float() / 255.0
        x = self.net(s)
        x = x.view(s.size(0), self.conv_out_dim)
        x = self.linear(x)
        return x

    @torch.jit.script_method
    def act(
        self, obs: Dict[str, torch.Tensor], hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self._conv_forward(obs["s"])
        # x: [batch, hid]
        x = x.unsqueeze(0)
        # x: [seq, batch, hid]
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        a = a.squeeze(0)
        # a: [batch, num_action]
        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return:
            q(s, a): [seq, batch] of given action sequence
            greedy_action: [seq, batch] that the current model would have taken
        """
        legal_move = obs["legal_move"]
        o, _ = self.unroll_rnn(obs, hid)

        a = self.fc_a(o)
        v = self.fc_v(o)
        q = self.duel(v, a, legal_move)
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        legal_q = (1 + q - q.min()) * legal_move
        greedy_action = legal_q.argmax(2).detach()
        return qa, greedy_action

    @torch.jit.script_method
    def unroll_rnn(self,
                   obs: Dict[str, torch.Tensor],
                   hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        s = obs["s"]
        assert s.dim() == 5  # [seq, batch, c, h, w]
        # print('s size:', s.size())
        seq, batch, c, h, w = s.size()
        s = s.view(seq * batch, c, h, w)
        x = self._conv_forward(s)
        x = x.view(seq, batch, self.hid_dim)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        # a = self.fc_a(o)
        # v = self.fc_v(o)
        # q = self.duel(v, a, legal_move)
        # qa = q.gather(2, action["a"].unsqueeze(2)).squeeze(2)
        return o, {"h0": h, "c0": c}
