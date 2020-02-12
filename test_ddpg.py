from collections import deque

import torch
from unityagents import UnityEnvironment
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from ddpg_agent import DDPGAgent
from xp_buffers import BufferType

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_current_weights(env, agent, brain_name, num_agents, n_episodes=5, train_mode=True):
    total_score = 0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)
        dones = [False] * num_agents
        i = 0
        while not np.any(dones):
            actions = agent.act(states, noise=False)
            env_info = env.step(actions)[brain_name]
            states = env_info.vector_observations
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished
            scores += rewards
            i+=1
        print(i)
        print("Evaluation episode {}, score {}".format(i_episode, np.mean(scores)))
        total_score += np.mean(scores) / n_episodes

    return total_score


class DDPGConfig:
    def __init__(self):
        self.critic_lr = 1e-3
        self.actor_lr = 1e-4
        self.buffer_type = BufferType.NORMAL
        self.buffer_size = int(1e5)  # replay buffer size
        self.batch_size = 64  # replay buffer size
        self.gamma = 0.95  # replay buffer size
        self.tau = 1e-3
        self.seed = 999
        self.update_every = 1
        self.warmup_steps = 200
        self.low_action = -1
        self.high_action = 1
        self.noise_start = 0.5
        self.noise_end = 0.1
        self.noise_steps = 300 * 50
        # self.std = nn.Parameter(torch.zeros(action_dim))

    def __str__(self):
        return "DDPG_actor_lr_{}_critic_lr_{}_batch_size_{}_batch_size_{}_noise_start_{}_noise_end _{}".format(
            self.actor_lr, self.critic_lr, self.batch_size, self.update_every, self.noise_start, self.noise_end)


def generate_grid_config():
    configs = []
    batch_sizes = [128]
    update_every = [1]
    noise_start = [0.2]
    critic_lr = [1e-4]
    actor_lr = [1e-3]
    for b in batch_sizes:
        for u in update_every:
            for n in noise_start:
                for a in actor_lr:
                    for c in critic_lr:
                        tmp = DDPGConfig()
                        tmp.batch_size = b
                        tmp.update_every = u
                        tmp.noise_start = n
                        tmp.actor_lr = a
                        tmp.critic_lr = c
                        configs.append(tmp)
    # tmp.n_steps = n
    return configs


if __name__ == "__main__":
    # env = UnityEnvironment(file_name='Reacher_20/Reacher.x86_64')
    env = UnityEnvironment(file_name='environments/Reacher_20/Reacher.x86_64')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    for config in generate_grid_config():
        agent = DDPGAgent(state_size, action_size, ddpg_config=config, num_agents=num_agents)
        agent.ddpg_actor_local.load_state_dict(torch.load('ddpg_weights/actor_{}.pth'.format(config)))
        agent.ddpg_critic_local.load_state_dict(torch.load('ddpg_weights/critic_{}.pth'.format(config)))
        evaluate_current_weights(env, agent, brain_name, num_agents, 5)
        # torch.save(agent.actor_critic.state_dict(), 'weights/{}.pth'.format(agent.ddpg_config))
        # agent.actor_critic.load_state_dict(torch.load('weights/{}.pth'.format(agent.ddpg_config)))
        # test_agent(env, agent, brain_name, num_agents)
