import gym
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque


class EnvSpace:
    def __init__(self, dim):
        self.high = [1 for _ in range(dim)]
        self.low = [-1 for _ in range(dim)]
        self.shape = (dim,)


class TestEnv:
    def __init__(self):
        self.observation_space = EnvSpace(4)
        self.action_space = EnvSpace(1)
        self.current_observation = [0 for _ in range(4)]


class ActorNet(torch.nn.Module):
    def __init__(self, n_observation, n_action):
        super(ActorNet, self).__init__()

        self.first_layer = torch.nn.Linear(in_features=n_observation, out_features=100)
        self.first_active = torch.nn.Tanh()

        self.second_layer = torch.nn.Linear(in_features=100, out_features=n_action)
        self.second_active = torch.nn.Tanh()

    def forward(self, observation):
        x = self.first_active(self.first_layer(observation))
        action = self.second_active(self.second_layer(x))
        return action


class CriticNet(torch.nn.Module):
    def __init__(self, n_observation, n_action):
        super(CriticNet, self).__init__()

        self.first_layer = torch.nn.Linear(in_features=(n_observation + n_action), out_features=100)
        self.first_active = torch.nn.Tanh()

        self.second_layer = torch.nn.Linear(in_features=100, out_features=1)

    def forward(self, observation, action):
        x = torch.cat((observation, action), dim=-1)

        x = self.first_active(self.first_layer(x))
        reward = self.second_layer(x)

        return reward


class ActorCriticAgent:
    def __init__(self, observation_space, action_space, calc_device='cpu'):
        self.tau = 0.001
        self.threshold = 0.0
        self.memory = deque(maxlen=2000)
        self.observation_space = observation_space
        self.action_space = action_space
        self.train_mode = True

        self.calc_device = calc_device
        self.actor_net = ActorNet(observation_space.shape[0], action_space.shape[0]).to(self.calc_device)
        self.critic_net = CriticNet(observation_space.shape[0], action_space.shape[0]).to(self.calc_device)
        self.actor_target = ActorNet(observation_space.shape[0], action_space.shape[0]).to(self.calc_device)
        self.critic_target = CriticNet(observation_space.shape[0], action_space.shape[0]).to(self.calc_device)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=0.01)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=0.01)
        self.actor_loss = torch.nn.MSELoss()
        self.critic_loss = torch.nn.MSELoss()

        self.hard_update(self.actor_target, self.actor_net)
        self.hard_update(self.critic_target, self.critic_net)

    def training(self, observations, rewards, targets=None):
        observe_tensor = torch.from_numpy(np.float32(observations)).to(self.calc_device)
        actions_batch = self.action(observations)
        actions_tensor = torch.from_numpy(np.float32(actions_batch)).to(self.calc_device)
        rewards_batch = -np.abs(targets - actions_batch)
        ret_loss = [0, 0]

        rewards_tensor = torch.from_numpy(np.float32(rewards_batch)).to(self.calc_device)

        self.critic_net.zero_grad()
        pred_y = self.critic_net(observe_tensor, actions_tensor)
        loss = self.critic_loss(pred_y, rewards_tensor)
        # print(loss.cpu())
        loss.backward()
        self.critic_optim.step()
        ret_loss[0] = loss.cpu().detach().numpy()

        self.actor_net.zero_grad()
        loss = -self.critic_net(observe_tensor, self.actor_net(observe_tensor))
        loss = loss.mean()
        loss.backward()
        self.actor_optim.step()
        ret_loss[1] = loss.cpu().detach().numpy()

        self.soft_update(self.actor_target, self.actor_net, self.tau)
        self.soft_update(self.critic_target, self.critic_net, self.tau)

        return ret_loss

    def action(self, observations):
        observe_inside = self.observes_map(observations)

        observe_tensor = torch.from_numpy(np.float32(observe_inside)).to(self.calc_device)
        actions_tensor = self.actor_target(observe_tensor)
        action_inside = actions_tensor.cpu().detach().numpy()

        if self.train_mode:
            for i in range(action_inside.shape[0]):
                if random.random() > self.threshold:
                    for col_idx in range(self.action_space.shape[0]):
                        action_inside[i, col_idx] = random.random() * 2 - 1

        action = self.action_unmap(action_inside)

        return action

    def critic(self, observations, actions):
        observe_inside = self.observes_map(observations)
        action_inside = self.action_map(actions)
        observe_tensor = torch.from_numpy(np.float32(observe_inside)).to(self.calc_device)
        action_tensor = torch.from_numpy(np.float32(action_inside)).to(self.calc_device)
        critic_tensor = self.critic_target(observe_tensor, action_tensor)

        return critic_tensor.cpu().detach().numpy()

    def remember(self, observe, action, reward, next_observe):
        self.memory.append((observe, action, reward, next_observe))

    @staticmethod
    def soft_update(target_net, source_net, tau):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    def hard_update(self, target_net, source_net):
        self.soft_update(target_net, source_net, 1)

    @staticmethod
    def value_map(area, value):
        """
        Map the value from area to [1, -1]
        :param area:
        :param value: shape should be n * 1
        :return:
        """
        ret_value = np.zeros([value.shape[0], 1])
        area_mid = (area[0] + area[1])
        area_len = (area[0] - area[1])
        for i in range(value.shape[0]):
            ret_value[i, 0] = (2 * value[i, 0] - area_mid) / area_len
        return ret_value

    @staticmethod
    def value_unmap(area, value):
        """
        Map the value from [1, -1] to area.
        :param area:
        :param value:
        :return:
        """
        ret_value = np.zeros([value.shape[0], 1])
        area_mid = (area[0] + area[1])
        area_len = (area[0] - area[1])
        for i in range(value.shape[0]):
            ret_value[i, 0] = (value[i, 0] * area_len + area_mid) / 2
        return ret_value

    def matrix_map(self, matrix, space):
        matrix_out = None
        for col_idx in range(space.shape[0]):
            if matrix_out is None:
                matrix_out = self.value_map([space.high[col_idx], space.low[col_idx]], matrix[:, col_idx:col_idx+1])
            else:
                matrix_tmp = self.value_map([space.high[col_idx], space.low[col_idx]], matrix[:, col_idx:col_idx+1])
                matrix_out = np.concatenate((matrix_out, matrix_tmp), axis=1)
        return matrix_out

    def matrix_unmap(self, matrix, space):
        matrix_out = None
        for col_idx in range(space.shape[0]):
            if matrix_out is None:
                matrix_out = self.value_unmap([space.high[col_idx], space.low[col_idx]], matrix[:, col_idx:col_idx+1])
            else:
                matrix_tmp = self.value_unmap([space.high[col_idx], space.low[col_idx]], matrix[:, col_idx:col_idx+1])
                matrix_out = np.concatenate((matrix_out, matrix_tmp), axis=1)
        return matrix_out

    def observes_map(self, observations):
        return self.matrix_map(observations, self.observation_space)

    def action_map(self, actions):
        return self.matrix_map(actions, self.action_space)

    def action_unmap(self, actions):
        return self.matrix_unmap(actions, self.action_space)

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

        actor_file = path + 'actor.pkl'
        critic_file = path + 'critic.pkl'

        torch.save(self.actor_net, actor_file)
        torch.save(self.critic_net, critic_file)


def main():
    if torch.cuda.is_available():
        calc_device = 'cuda'
    else:
        calc_device = 'cpu'
    print('Use {}'.format(calc_device))
    env = gym.make('MountainCarContinuous-v0')
    # env = TestEnv()
    agent = ActorCriticAgent(env.observation_space, env.action_space, calc_device)

    plt.ion()
    plt.show()
    plt.cla()

    for i in range(5):
        done = False
        reward_list = []
        position_list = []
        observe = env.reset()

        while not done:
            env.render()
            action = agent.action(np.array([observe]))
            next_observe, reward, done, _ = env.step(action[0])
            print("{}, {}, {}, {}".format(observe, action, reward, done))
            agent.remember(observe, action, reward, next_observe)

            observe = next_observe

            reward_list.append(reward)
            position_list.append(observe[0])
            plt.cla()
            plt.plot(reward_list, 'b-')
            plt.plot(position_list, 'r-')
            # plt.pause(0.01)

        # plt.cla()
        # plt.plot(result_list, 'g-')
        # plt.plot(target_list, 'b-')
        # plt.plot(action_list, 'r-')
        # plt.plot(critic_list, 'g*')
        # plt.pause(0.001)


if __name__ == '__main__':
    main()
