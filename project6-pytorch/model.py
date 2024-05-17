
"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""

from torch.nn import Module
from torch.nn import  Linear
from torch import tensor, double, optim
from torch.nn.functional import relu, mse_loss



class DeepQNetwork(Module):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim
        super(DeepQNetwork, self).__init__()
        # Remember to set self.learning_rate, self.numTrainingGames,
        # and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.15
        self.numTrainingGames = 1700
        self.batch_size = 512

        hidden_size = 500
        self.inputLayer = Linear(self.state_size, hidden_size)
        self.hiddenLayer1 = Linear(hidden_size, hidden_size)
        # self.hiddenLayer2 = Linear(hidden_size, hidden_size)
        # self.hiddenLayer3 = Linear(hidden_size, hidden_size)
        self.outputLayer = Linear(hidden_size, self.num_actions)

        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        "**END CODE"""
        self.double()


    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        predect_Q = self.forward(states)
        return mse_loss(predect_Q, Q_target)


    def forward(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        layer1Result = self.inputLayer(states)
        layer2Result = self.hiddenLayer1(relu(layer1Result))
        # layer3Result = self.hiddenLayer2(relu(layer2Result))
        # layer4Result = self.hiddenLayer3(relu(layer3Result))
        return self.outputLayer(relu(layer2Result))

    
    def run(self, states):
        return self.forward(states)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        You can look at the ML project for an idea of how to do this, but note that rather
        than iterating through a dataset, you should only be applying a single gradient step
        to the given datapoints.

        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        self.optimizer.zero_grad()
        loss = self.get_loss(states, Q_target)
        loss.backward()
        self.optimizer.step()
         