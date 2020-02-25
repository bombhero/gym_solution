import gym
import torch
import numpy as np
import matplotlib.pyplot as plt


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
        action = self.second_layer(x)
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
    def __init__(self, n_observation, n_action, calc_device='cpu'):
        self.tau = 0.001

        self.calc_device = calc_device
        self.actor_net = ActorNet(n_observation, n_action).to(self.calc_device)
        self.critic_net = CriticNet(n_observation, n_action).to(self.calc_device)
        self.actor_target = ActorNet(n_observation, n_action).to(self.calc_device)
        self.critic_target = CriticNet(n_observation, n_action).to(self.calc_device)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=0.01)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=0.01)
        # self.actor_optim = torch.optim.SGD(self.actor_net.parameters(), lr=0.01)
        # self.critic_optim = torch.optim.SGD(self.actor_net.parameters(), lr=0.01)
        # self.actor_optim = torch.optim.Adagrad(self.actor_net.parameters(), lr=0.01)
        # self.critic_optim = torch.optim.Adagrad(self.actor_net.parameters(), lr=0.01)
        self.actor_loss = torch.nn.MSELoss()
        self.critic_loss = torch.nn.MSELoss()

        self.hard_update(self.actor_target, self.actor_net)
        self.hard_update(self.critic_target, self.critic_net)

    def training(self, observations, rewards, targets=None):
        observe_tensor = torch.from_numpy(np.float32(observations)).to(self.calc_device)
        actions_tensor = self.actor_target(observe_tensor)
        actions_batch = actions_tensor.cpu().detach().numpy()
        rewards_batch = -np.abs(targets - actions_batch)

        # for i in range(rewards_batch.shape[0]):
        #     if rewards_batch[i] > 0:
        #         rewards_batch[i] = 1 / rewards_batch[i]
        rewards_tensor = torch.from_numpy(np.float32(rewards_batch)).to(self.calc_device)

        self.critic_net.zero_grad()
        pred_y = self.critic_net(observe_tensor, actions_tensor)
        loss = self.critic_loss(pred_y, rewards_tensor)
        # print(loss.cpu())
        loss.backward()
        self.critic_optim.step()

        self.actor_net.zero_grad()
        loss = -self.critic_net(observe_tensor, self.actor_net(observe_tensor))
        loss = loss.mean()
        loss.backward()
        self.actor_optim.step()

        self.soft_update(self.actor_target, self.actor_net, self.tau)
        self.soft_update(self.critic_target, self.critic_net, self.tau)

    # def training(self, observations, rewards, targets):
    #     observe_tensor = torch.from_numpy(np.float32(observations)).to(self.calc_device)
    #     targets_tensor = torch.from_numpy(np.float32(targets)).to(self.calc_device)
    #
    #     self.actor_net.train()
    #     self.actor_optim.zero_grad()
    #     pred_y = self.actor_net(observe_tensor)
    #     loss = self.actor_loss(pred_y, targets_tensor)
    #     loss.backward()
    #     self.actor_optim.step()

    def action(self, observations):
        observe_tensor = torch.from_numpy(np.float32(observations)).to(self.calc_device)
        actions_tensor = self.actor_target(observe_tensor)

        return actions_tensor.cpu().detach().numpy()

    def critic(self, observations, actions):
        observe_tensor = torch.from_numpy(np.float32(observations)).to(self.calc_device)
        action_tensor = torch.from_numpy(np.float32(actions)).to(self.calc_device)
        critic_tensor = self.critic_target(observe_tensor, action_tensor)

        return critic_tensor.cpu().detach().numpy()


    @staticmethod
    def soft_update(target_net, source_net, tau):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    def hard_update(self, target_net, source_net):
        self.soft_update(target_net, source_net, 1)


def main():
    if torch.cuda.is_available():
        calc_device = 'cuda'
    else:
        calc_device = 'cpu'
    # env = gym.make('MountainCarContinuous-v0')
    env = TestEnv()
    agent = ActorCriticAgent(env.observation_space.shape[0],
                             env.action_space.shape[0],
                             calc_device)

    plt.ion()
    plt.show()
    plt.cla()
    target_list = []
    action_list = []
    result_list = []
    critic_list = []
    observe_test = np.random.random([1, 4]) * 2 - 1
    target_test = sum(observe_test[0]) / 4

    for i in range(500):
        observe_batch = np.random.random([128, 4]) * 2 - 1
        target_batch = np.zeros([128, 1])

        for j in range(target_batch.shape[0]):
            target_batch[j, 0] = sum(observe_batch[j]) / 4

        for _ in range(200):
            agent.training(observe_batch, None, target_batch)

        action_test = agent.action(observe_test)
        critic_test = agent.critic(observe_test, action_test)
        # print("{}, {}".format(target_test, action_test[0, 0]))
        target_list.append(target_test)
        action_list.append(action_test[0, 0])
        critic_list.append((critic_test[0, 0]))
        result_list.append(-abs(target_test - action_test[0, 0]))
        print(result_list[-1])
        plt.cla()
        plt.plot(result_list, 'g-')
        plt.plot(target_list, 'b-')
        plt.plot(action_list, 'r-')
        plt.plot(critic_list, 'g*')
        plt.pause(0.001)

    plt.pause(10)


if __name__ == '__main__':
    main()
