import torch
import torch.nn as nn

class UrbanPlanningReward(nn.Module):
    def __init__(self, cfg, agent, shared_net):
        super(UrbanPlanningReward, self).__init__()
        self.agent = agent
        self.shared_net = shared_net
        self.reward_head = self.create_reward_head(
            self.shared_net.output_policy_land_use_size, cfg['reward_head_hidden_size']
        )        
    def create_reward_head(self, input_size, hidden_size):
        reward_head = nn.Sequential()
        for i in range(len(hidden_size)):
            if i == 0:
                reward_head.add_module(
                    'reward_linear_{}'.format(i),
                    nn.Linear(input_size, hidden_size[i])
                )
            else:
                reward_head.add_module(
                    'reward_linear_{}'.format(i),
                    nn.Linear(hidden_size[i - 1], hidden_size[i], bias=False)
                )
            if i < len(hidden_size) - 1:
                reward_head.add_module(
                    'reward_tanh_{}'.format(i),
                    nn.Tanh()
                )
        return reward_head

    def forward(self, x):
        with torch.no_grad():
            state, _, _, _, _, stage = self.shared_net(x)
        reward = self.reward_head(state)
        return reward