import torch
import torch.nn as nn


class Policy(nn.Module):
    """
    Policy network for urban planning.
    """
    def __init__(self, cfg, agent, shared_net):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.shared_net = shared_net
        self.policy_road_head = self.create_policy_head(self.shared_net.output_policy_road_size, cfg['policy_road_head_hidden_size'], 'road')

    def create_policy_head(self, input_size, hidden_size, name):
        """Create the policy land_use head."""
        policy_head = nn.Sequential()
        for i in range(len(hidden_size)):
            if i == 0:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(input_size, hidden_size[i])
                )
            else:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(hidden_size[i - 1], hidden_size[i], bias=False)
                )
            if i < len(hidden_size) - 1:
                policy_head.add_module(
                    '{}_tanh_{}'.format(name, i),
                    nn.Tanh()
                )
            elif hidden_size[i] == 1:
                policy_head.add_module(
                    '{}_flatten_{}'.format(name, i),
                    nn.Flatten()
                )
        return policy_head

    def forward(self, x):
        policy_road, _, mask, stage = self.shared_net(x)

        if stage[:, 1].sum() == 0:
            logits = self.policy_road_head(policy_road)
            paddings = torch.ones_like(mask, dtype=self.agent.dtype)*(-2.**32 + 1)
            masked_logits = torch.where(mask.bool(), logits, paddings)
            dist = torch.distributions.Categorical(logits=masked_logits)
        else:
            dist = None

        return dist, stage
        

    def select_action(self, x, mean_action=False):
        
        dist, stage = self.forward(x)
        batch_size = stage.shape[0]
        action = torch.zeros(batch_size, 1, dtype=self.agent.dtype, device=stage.device)
        if mean_action:
            road_action = dist.probs.argmax(dim=1).to(self.agent.dtype)
        else:
            road_action = dist.sample().to(self.agent.dtype)
        action = road_action
        return action

    def get_log_prob_entropy(self, x, action):
        dist, stage = self.forward(x)
        batch_size = stage.shape[0]
        log_prob = torch.zeros(batch_size, dtype=self.agent.dtype, device=stage.device)
        entropy = torch.zeros(batch_size, dtype=self.agent.dtype, device=stage.device)

        road_action = action

        road_log_prob = dist.log_prob(road_action)
        log_prob = road_log_prob
        entropy = dist.entropy()

        return log_prob.unsqueeze(1), entropy.unsqueeze(1)
