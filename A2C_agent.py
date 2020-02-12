import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import ActorCriticModel
from xp_buffers import RollOutMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CAgent:

    def __init__(self, a2c_config, num_agents, state_dim, action_dim):
        self.config = a2c_config
        self.roll_out_length = a2c_config.n_steps
        self.actor_critic = ActorCriticModel(state_dim, action_dim, a2c_config.seed, a2c_config.actor_hidden,
                                             a2c_config.critic_hidden)
        self.roll_out_memory = RollOutMemory()
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.config.learning_rate)
        self.n_steps = a2c_config.n_steps
        self.gradient_clip = a2c_config.gradient_clip
        self.num_agents = num_agents
        self.seed = random.seed(a2c_config.seed)

    def clean_buffer(self):
        self.roll_out_memory.clear()

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        action, _, _, _  = self.actor_critic.forward(state)
        return action.cpu().numpy()

    def eval_state(self, state):
        state = torch.from_numpy(state).float().to(device)
        actions, log_prob, entropy, state_value = self.actor_critic.forward(state)
        return actions, log_prob, entropy, state_value

    def step(self, actions, rewards, log_prob, entropy, not_dones, state_values):
        self.roll_out_memory.add(actions, rewards, log_prob, entropy, not_dones, state_values)

        if self.roll_out_memory.size() >= self.n_steps:
            loss = self.learn()
            self.clean_buffer()
            return loss
        return None

    def prepare_experiences(self, experiences):

        processed_experience = [None] * (len(experiences) - 1)
        _advantage = torch.tensor(np.zeros((self.num_agents, 1))).float().to(device=device)  # initialize advantage Tensor
        _return = experiences[-1][-1].detach()
        for i in range(len(experiences) - 2, -1, -1):
            _action, _reward, _log_prob, _entropy, _not_done, _value = experiences[i]
            _not_done = torch.tensor(_not_done, device=device).unsqueeze(1).float()
            _reward = torch.tensor(_reward, device=device).unsqueeze(1)
            _next_value = experiences[i + 1][-1]
            _return = _reward + self.config.gamma * _not_done * _return
            if not self.config.use_gae:
                _advantage = _reward + self.config.gamma * _not_done * _next_value.detach() - _value.detach()
            else:
                td_error = _reward + self.config.gamma * _not_done * _next_value.detach() - _value.detach()
                _advantage = _advantage * self.config.gae_tau * self.config.gamma * _not_done + td_error
            processed_experience[i] = [_log_prob, _value, _return, _advantage, _entropy]

        log_prob, value, returns, advantages, entropy = map(lambda x: torch.cat(x, dim=0), zip(*processed_experience))

        return log_prob, value, returns, advantages, entropy


    def learn(self):

        experiences = self.roll_out_memory.experiences

        log_prob, value, returns, advantages, entropy = self.prepare_experiences(experiences)

        policy_loss = -(log_prob * advantages)
        value_loss = 0.5 * (returns - value).pow(2)
        entropy_loss = entropy

        total_loss = (policy_loss - self.config.entropy_weight * entropy_loss +
                      self.config.value_loss_weight * value_loss)

        self.optimizer.zero_grad()
        total_loss.mean().backward()
        nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        return total_loss
