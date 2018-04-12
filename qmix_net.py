import torch
import torch.nn as nn
import torch.nn.functional as F


class QMIXNet(nn.Module):
    """
    This is the end-to-end differentiable network which learns
    values for the centralized Q network (with decentralized actors
    from agent networks)
    """
    def __init__(self,
                 num_agents,
                 action_space,
                 state_shape,
                 agent_shape,
                 agent_hidden_size,
                 mixing_hidden_size):
        """

        :param num_agents: Number of agents
        :param action_space: Size of the action space
        :param state_shape: Shape of the global state tensor
        :param agent_shape: Shape of the agent local observation tensor
        :param agent_hidden_size: Shape of the hidden state of agent observation in GRU
        :param mixing_hidden_size: Size of hidden layer of the Mixing Network
        """
        super(QMIXNet, self).__init__()

        self.num_agents = num_agents
        self.action_space = action_space
        self.state_shape = state_shape
        self.agent_shape = agent_shape
        self.agent_hidden_size = agent_hidden_size
        self.mixing_hidden_size = mixing_hidden_size

        # Agent Network, a GRU with one Linear layer on both ends
        self.agent_ff_in = nn.Linear(self.agent_shape, self.agent_shape)
        self.agent_net = nn.GRU(self.agent_shape, self.agent_hidden_size)
        self.agent_ff_out = nn.Linear(self.agent_hidden_size, self.action_space)

        # Used to generate mixing network
        self.hyper_net1 = nn.Linear(self.state_shape, self.num_agents * self.mixing_hidden_size)
        self.hyper_net2 = nn.Linear(self.state_shape, self.mixing_hidden_size)

    def forward(self, global_state, agent_obs):
        """
        The forward model takes the state to be given to the hyper-networks
        and agent observations as a single tensor (concatenation of
        agent local current observation, one-hot encoded last action, one-hot encoded agent_id)
        :param global_state: state_shape
        :param agent_obs: num_agents x agent_shape
        :return: qtot
        """

        # Agent q values
        q_n = self.agent_ff_in(agent_obs)
        q_n = self.agent_net(q_n)
        q_n = self.agent_ff_out(q_n).max(dim=1)[0]

        # Weights for the Mixing Network (absolute for monotonicity)
        w1 = self.hyper_net1(global_state).abs()
        w2 = self.hyper_net2(global_state).abs()

        # Reshape for Mixing Network
        w1 = w1.view(self.num_agents, self.mixing_hidden_size)
        w2 = w2.view(self.mixing_hidden_size, 1)

        # Calculate mixing of agent values for q_tot
        q_tot = F.elu(torch.mm(q_n, w1))
        q_tot = F.elu(torch.mm(q_tot, w2))

        return q_tot
