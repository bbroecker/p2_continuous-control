import json
import numpy as np
import matplotlib.pyplot as plt

json_folder = "../jsons/"
json_ddpg_folder = "../ddpg_json/"
json_a2c_folder = "../a2c_json/"
figure_folder = "/home/broecker/src/udacity_rl/report_p2/figures/"

data_ddpg_actor_lr_0_001_critic_lr_0_0001 = {
    "file_name": "DDPG_actor_lr_0.001_critic_lr_0.0001_batch_size_128.json",
    "label": "DDPG_actor_lr_1e-3_critic_lr_1e-4_batch_size_128"}

data_ddpg_actor_lr_0_001_critic_lr_0_0001_long = {
    "file_name": "DDPG_actor_lr_0.001_critic_lr_0.001_batch_size_128_long.json",
    "label": "DDPG_actor_lr_1e-3_critic_lr_1e-4_batch_size_128"}

data_ddpg_actor_lr_0_0001_critic_lr_0_001 = {
    "file_name": "DDPG_actor_lr_0.0001_critic_lr_0.001_batch_size_128.json",
    "label": "DDPG_actor_lr_1e-4_critic_lr_1e-3_batch_size_128"}

DDPG_actor_lr_0_001_critic_lr_0_0001_batch_size_64 = {
    "file_name": "DDPG_actor_lr_0.001_critic_lr_0.0001_batch_size_64.json",
    "label": "DDPG_actor_lr_1e-3_critic_lr_1e-4_batch_size_64"}

DDPG_actor_lr_0_001_critic_lr_0_0001_batch_size_32 = {
    "file_name": "DDPG_actor_lr_0.001_critic_lr_0.0001_batch_size_32.json",
    "label": "DDPG_actor_lr_1e-3_critic_lr_1e-4_batch_size_32"}

A2C_n_steps_3 = {
    "file_name": "A2C_n_steps_3.json",
    "label": "A2C n_steps=3"}

A2C_n_steps_4 = {
    "file_name": "A2C_n_steps_4.json",
    "label": "A2C n_steps=4"}

A2C_n_steps_5 = {
    "file_name": "A2C_n_steps_5.json",
    "label": "A2C n_steps=5"}

A2C_n_steps_7 = {
    "file_name": "A2C_n_steps_7.json",
    "label": "A2C n_steps=7"}

A2C_n_steps_5_gae = {
    "file_name": "A2C_n_steps_5_gae_0.95.json",
    "label": "A2C_GAE n_steps=5 λ=0.95"}

def plot_reward_per_step(data_set, folders, colors, save_name, x_limit=100, y_limit=60, aspect=1):
    labels = []
    for data_frame, color, folder in zip(data_set, colors, folders):
        labels.append(data_frame["label"])
        f = open(folder + data_frame["file_name"], 'r')

        data = np.array(json.load(f))
        xs = data[:, 1]
        ys = data[:, 2]
        firsts = [x for x, y in zip(xs, ys) if y > 13.]
        first = 'n/a' if not firsts else firsts[0]
        print("{} & {} &{:.2f} \\\\".format(data_frame["label"], first, max(ys)))
        print("\\hline")
        plt.plot(xs, ys, color=color, label=data_frame["label"], linewidth=1)
        f.close()
    plt.grid(color='#7f7f7f', linestyle='-', linewidth=1)
    plt.hlines(30, 0, x_limit, linestyles='dashed', linewidth=2.5)
    plt.xlim([0., x_limit])
    plt.ylim([0., y_limit])
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.axes().set_aspect(aspect=aspect)
    plt.savefig(figure_folder + save_name, format="pdf", pad_inches=0, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_reward_per_step([data_ddpg_actor_lr_0_001_critic_lr_0_0001, data_ddpg_actor_lr_0_0001_critic_lr_0_001,
                          DDPG_actor_lr_0_001_critic_lr_0_0001_batch_size_64,
                          DDPG_actor_lr_0_001_critic_lr_0_0001_batch_size_32], [json_ddpg_folder] * 4, ['r', 'b', 'g', 'c'],
                         'ddpg.pdf')

    plot_reward_per_step([data_ddpg_actor_lr_0_001_critic_lr_0_0001_long], [json_ddpg_folder] * 1, ['r'],
                         'ddpg_long.pdf', y_limit=70, aspect=0.75, x_limit=160)

    plot_reward_per_step([A2C_n_steps_3, A2C_n_steps_4, A2C_n_steps_5], [json_a2c_folder] * 3, ['r', 'b', 'g'],
                         'a2c_tests.pdf', x_limit=200, y_limit=70)

    plot_reward_per_step([A2C_n_steps_5_gae], [json_a2c_folder] * 1, ['r'],
                         'a2c_gae.pdf', x_limit=200)


    # plot_reward_per_step([data_double_dqn, data_double_dqn_skip], ['r', 'b'], '../figures/priory.pdf')
