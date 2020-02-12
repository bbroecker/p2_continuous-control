from enum import Enum

import numpy as np
import random

from models import  DDPGActorModel, DDPGCriticModel

import torch
import torch.optim as optim

from xp_buffers import BufferType, StandardBuffer, PriorityBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, ddpg_config):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(ddpg_config.seed)
        self.ddpg_config = ddpg_config
        self.num_agents = num_agents
        seed = ddpg_config.seed
        # self.skip_frames = ddpg_config.skip_frames

        self.ddpg_actor_local = DDPGActorModel(state_dim=state_size, action_dim=action_size, seed=seed).to(device)
        self.ddpg_actor_target = DDPGActorModel(state_dim=state_size, action_dim=action_size, seed=seed).to(device)

        self.ddpg_critic_local = DDPGCriticModel(state_dim=state_size, action_dim=action_size, seed=seed).to(device)
        self.ddpg_critic_target = DDPGCriticModel(state_dim=state_size, action_dim=action_size, seed=seed).to(device)

        self.exploration_noise = OrnsteinUhlenbeckProcess(size=(self.action_size,), std=LinearSchedule(ddpg_config))

        self.critic_optimizer = optim.Adam(self.ddpg_critic_local.parameters(), lr=ddpg_config.critic_lr)
        self.actor_optimizer = optim.Adam(self.ddpg_actor_local.parameters(), lr=ddpg_config.actor_lr)

        self.learning_steps = 0

        # self.criterion = nn.MSELoss()
        # Replay memory
        if ddpg_config.buffer_type == BufferType.NORMAL:
            self.buffer = StandardBuffer(action_size, ddpg_config)
        elif ddpg_config.buffer_type == BufferType.PRIORITY:
            self.buffer = PriorityBuffer(action_size, ddpg_config)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.gamma = ddpg_config.gamma
        self.tau = ddpg_config.tau
        self.update_every = ddpg_config.update_every
        self.batch_size = ddpg_config.batch_size

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience per agent in the replay buffer
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            action_dev = torch.from_numpy(action).float().unsqueeze(0).to(device)
            state_dev = torch.from_numpy(state).float().unsqueeze(0).to(device)
            next_state_dev = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
            next_action = self.ddpg_actor_target.forward(next_state_dev).detach().to(device)

            state_action_value = self.ddpg_critic_target.forward(next_state_dev, next_action).detach()
            q_new = reward + self.gamma * state_action_value * (1 - done)

            q_old = self.ddpg_critic_local.forward(state_dev, action_dev).data.cpu().numpy()[0]
            error = abs(q_new.cpu().detach().numpy() - q_old)

            self.buffer.add((state, action, reward, next_state, done), error)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.buffer) > self.batch_size:
                return self.learn()
        return None, None

    def buffer_size(self):
        return len(self.buffer)

    def reset(self):
        self.learning_steps = 0

    def act(self, state, noise=False):
        """
        Action seletection process used during training
        :param state:
        :param noise:
        :return:
        """
        state = torch.from_numpy(state).float().to(device)
        action = self.ddpg_actor_local.forward(state).cpu().detach().numpy()
        if not noise:
            return action
        elif self.learning_steps < self.ddpg_config.warmup_steps:
            action = np.random.uniform(self.ddpg_config.low_action, self.ddpg_config.high_action, (self.num_agents, self.action_size))
        else:
            action += self.exploration_noise.sample()
            # action = action.detach().cpu().numpy()
        self.learning_steps += 1
        return np.clip(action, self.ddpg_config.low_action, self.ddpg_config.high_action)


    def learn(self):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        experiences, idxs, is_weights = self.buffer.sample()
        states, actions, rewards, next_states, dones = experiences

        next_actions = self.ddpg_actor_target.forward(next_states).detach()
        q_next = self.ddpg_critic_target.forward(next_states, next_actions).detach()

        q_targets = rewards + (self.gamma * q_next * (1 - dones))
        q_expected = self.ddpg_critic_local.forward(states, actions)

        errors = torch.abs(q_targets - q_expected).data.cpu().numpy()
        for idx, error in zip(idxs, errors):
            self.buffer.update(idx, error)

        self.critic_optimizer.zero_grad()
        # if prioritised buffer is active the weights will effect the update
        # loss = (torch.FloatTensor(is_weights).to(device) * F.mse_loss(q_expected, q_targets).squeeze())
        # loss = loss.mean()
        critic_loss = (q_targets - q_expected).pow(2).mul(0.5).mean()
        critic_loss.backward()
        critic_loss_output = critic_loss.cpu().detach().numpy()
        self.critic_optimizer.step()

        actions = self.ddpg_actor_local.forward(states)
        # policy_loss = -(torch.FloatTensor(is_weights).to(device) * self.ddpg_critic_local.forward(states, actions))
        policy_loss = -self.ddpg_critic_local.forward(states, actions).mean()
        # policy_loss = policy_loss.mean()
        self.actor_optimizer.zero_grad()
        # if prioritised buffer is active the weights will effect the update

        policy_loss.backward()
        self.actor_optimizer.step()
        actor_loss_output = policy_loss.cpu().detach().numpy()

        # ------------------- update target network ------------------- #
        self.soft_update()

        return critic_loss_output, actor_loss_output

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.ddpg_critic_target.parameters(), self.ddpg_critic_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        for target_param, local_param in zip(self.ddpg_actor_target.parameters(), self.ddpg_actor_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class OrnsteinUhlenbeckProcess:
    """
    Implementation inspiration by: https://github.com/ShangtongZhang/DeepRL
    """

    def __init__(self, size, std, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.x_prev = None
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std() * np.sqrt(
            self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class LinearSchedule:
    """
    Implementation inspiration by: https://github.com/ShangtongZhang/DeepRL
    Class to automatically increase/decrease the noise std-variation
    """
    def __init__(self, ddpp_config):
        start = ddpp_config.noise_start
        end = ddpp_config.noise_end
        steps = ddpp_config.noise_steps
        if end is None:
            end = ddpp_config.start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val