import tensorflow
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
import csv
import pandas as pd
import sys
from torch.utils.tensorboard import SummaryWriter
from A2C import *
from func import *
import math
import matplotlib.pyplot as plt
from torch.utils import tensorboard
from collections import deque

if __name__ == "__main__":
    file_name = input("Enter file_name, window_size, Episode_count")
    window_size = input()
    episode_count = input()
    file_name = str(file_name)
    window_size = int(window_size)
    episode_count = int(episode_count)
    
# A2C hyperparameters
hidden_size = 256
learning_rate = 0.001
gamma = 0.99

# Initialize the environment
num_inputs = window_size
num_actions = 3
data = getCaseDataVec(file_name)
recovery_period = 14 # days
l = len(data) - 1

# Initialize the actor-critic network and optimizer
ac_model = ActorCritic(num_inputs, num_actions, hidden_size)
optimizer = optim.Adam(ac_model.parameters(), lr=learning_rate)

# Initialize dictionaries to store counts of each action at each time step
action_counts = {
    "Sit": 0,
    "Lockdown/Quarantine": 0,
    "Lockdown + Cure": 0
}
    
# Training loop
episode_rewards = []

for i in range(episode_count):
    state = getState(data, 0, window_size + 1)
    done = False
    episode_reward = 0
        
    while not done:
        action_probs, state_value = ac_model(t(state))
        dist = torch.distributions.Categorical(action_probs)
        print("Episode " + str(i) + "/" + str(episode_count))
        state = torch.FloatTensor(state)
        total_cases = 0
        totalinventory = []
        
        for k in range(l):
            action = dist.sample()
            # sit
            next_state = getState(data, k + 1, window_size + 1) #array
            reward = 0
            if action == 0: # sit
                totalinventory.append(data[k])
                total_cases = data[k]
                action_counts["Sit"] += 1
                reward = 0
                print("No control: " + formatCases(data[k]))
            elif action == 1 and len(totalinventory) > 0: # lockdown/quarantine
                oldest_cases = totalinventory.pop(0)
                recovered = 0
                if k >= recovery_period:
                    recovered = oldest_cases
                    action_counts["Lockdown/Quarantine"] += 1
                reward = max(recovered, 0)
                total_cases = data[k] - recovered # at time t
                print("Decrease due to lockdown: " + formatCases(total_cases) + " | Cases Reduced: " + formatCases(recovered))
            elif action == 2 and len(totalinventory) > 0 and data[k]>100:  # 100% effective cure given to 10% of infected + lockdown
                recovered_cured = 0
                cured  = 0
                action_counts["Lockdown + Cure"] += 1
                if k >= recovery_period:
                    recovered_cured = oldest_cases + 0.1 * data[k]
                    reward = max(recovered_cured, 0)
                    cases = data[k] - recovered_cured # at time t
                    total_cases = max(cases, 0)
                    print("Decrease due to lockdown and cure: " + formatCases(total_cases) + " | Cases Reduced: " + formatCases(recovered_cured))
                else:
                    cured = 0.1 * data[k]
                    reward = max(cured, 0)
                    total_cases = data[k] - cured
                    print("Decrease due to cure: " + formatCases(total_cases) + " | Cases Reduced: " + formatCases(cured))
            # Store the total cases reduced in the current episode in the cumulative_reduction list
            prob_act = dist.log_prob(action)
            done = True if k == l - 1 or total_cases == 0 else False
            state = next_state #array
            
            if done:
                print("--------------------------------")
                print("Total Cases: " + formatCases(total_cases))
                print("--------------------------------")


        # Compute the advantage
        next_state_value = ac_model(torch.FloatTensor(next_state))[1]
        delta = reward + gamma * next_state_value - state_value

        # Compute the loss
        actor_loss = -torch.log(action_probs[action]) * delta
        critic_loss = delta.pow(2)
        total_loss = actor_loss + critic_loss

        # Update the model
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        episode_reward += reward
        state = next_state
        
    episode_rewards.append(episode_reward)
    print(episode_rewards)
    
    plt.figure(figsize=(8, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards per Episode')
    plt.savefig('a2crep.png')
    
    # Plot the frequency of each action over time step 'n'
    actions = list(action_counts.keys())
    frequencies = list(action_counts.values())
    plt.figure(figsize=(6, 8))
    plt.bar(actions, frequencies)
    plt.xlabel('Actions')
    plt.ylabel('Frequency')
    plt.title('Frequency of Each Action over Time Step n')
    plt.savefig('a2cfrequencies.png')

    print(f"Episode {i + 1}, Reward: {episode_reward}")
