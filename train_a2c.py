from collections import deque

import torch
from unityagents import UnityEnvironment
import numpy as np

from A2C_agent import A2CAgent
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_current_weights(env, agent, brain_name, num_agents, n_episodes=5, train_mode=True):

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)
        dones = [False] * num_agents
        while not np.any(dones):
            actions, log_prob, entropy, state_value = agent.eval_state(states)
            env_info = env.step(actions.detach().cpu().numpy())[brain_name]
            states = env_info.vector_observations
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done # see if episode has finished
            scores += rewards

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores)))

    return scores


def train_agent(env, agent, brain_name, num_agents, weight_dir, n_episodes=1600, max_t=1000, summary_writer=None):

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=int(100. / num_agents)) # last 100 scores

    # save weight when the score is higher that current max
    max_score = 30.
    max_score_episode = 0

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)
        losses = []
        dones = [False] * num_agents
        while not np.any(dones):
            actions, log_prob, entropy, state_value = agent.eval_state(states)
            env_info = env.step(actions.detach().cpu().numpy())[brain_name]
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done # see if episode has finished
            not_dones = [1 - done for done in dones]
            # next_state, reward, done, _ = env.step(action)
            loss = agent.step(actions, rewards, log_prob, entropy, not_dones, state_value)
            if loss is not None:
                losses.append(loss.cpu().detach().numpy())
            states = next_states
            scores += rewards

        agent.clean_buffer()
        scores_window.append(np.mean(scores))  # save most recent score

        mean_score = np.mean(scores_window)
        if mean_score > max_score + 0.3:
            max_score = mean_score
            torch.save(agent.actor_critic.state_dict(), weight_dir)
            max_score_episode = i_episode - 100
            print('\nNew max score. Weights saved {:d} episodes!\tAverage Score: {:.2f}'.format(max_score_episode,
                                                                                                mean_score))

        if summary_writer is not None:
            summary_writer.add_scalar('Avg_Loss', np.mean(losses), i_episode)
            summary_writer.add_scalar('Avg_Reward', np.mean(scores_window), i_episode)

        print('\rEpisode {}\tAverage Score: {:.2f} Losses: {:.2f}'.format(i_episode, np.mean(scores),
                                                                          np.mean(losses)))
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} Losses: {:.2f}'.format(i_episode, np.mean(scores_window),
                                                                              np.mean(losses)))

    return scores


class A2CConfig:
    def __init__(self):
        self.learning_rate = 2e-4
        self.gradient_clip = 5
        self.n_steps = 5
        self.gamma = 0.95
        self.use_gae = True
        self.gae_tau = 0.95
        self.entropy_weight = 0.05
        self.value_loss_weight = 1.0
        self.seed = 999
        self.actor_hidden = [128, 128]
        self.critic_hidden = [128, 128]
        # self.std = nn.Parameter(torch.zeros(action_dim))

    def __str__(self):
        gae = "" if not self.use_gae else "gae_{}".format(self.gae_tau)
        return "A2C_n_steps_{}_{}".format(self.n_steps, gae)

def generate_grid_config():
    configs = []
    use_gae = [True]
    n_steps = [5]
    for n in n_steps:
        for gau in use_gae:
            tmp = A2CConfig()
            tmp.use_gae = gau
            tmp.n_steps = n
            configs.append(tmp)
    return configs


def load_weights(agent, weight_dir):
    agent.actor_critic.load_state_dict(torch.load(weight_dir))


if __name__== "__main__":
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

        agent = A2CAgent(config, num_agents, state_size, action_size)
        weight_dir = 'a2c_weights/{}.pth'.format(config)
        train_agent(env, agent, brain_name, num_agents, weight_dir=weight_dir, summary_writer=SummaryWriter("a2c_logs/{}".format(config)), n_episodes=200)
        # torch.save(agent.actor_critic.state_dict(), 'weights/{}.pth'.format(agent.config))
        # agent.actor_critic.load_state_dict(torch.load('weights/{}.pth'.format(agent.config)))
        # test_agent(env, agent, brain_name, num_agents)