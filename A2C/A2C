import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import tensorboard
from func import *

# helper function to convert numpy arrays to tensors
def t(x): return torch.tensor(x[0], dtype=torch.float32) #tensor

# Define the A2C network
class ActorCritic(nn.Module): #one network for both
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc_actor = nn.Linear(hidden_size, num_actions)
        self.fc_critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.fc_actor(x), dim=-1)
        state_value = self.fc_critic(x)
        return action_probs, state_value
