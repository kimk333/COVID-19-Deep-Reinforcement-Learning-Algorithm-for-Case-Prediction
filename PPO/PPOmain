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
from PPO import *
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
    
# config
state_dim = window_size
n_actions = 3
data = getCaseDataVec(file_name)
recovery_period = 14 # days
l = len(data) - 1
actor = Actor(state_dim, n_actions, activation=Mish)
critic = Critic(state_dim, activation=Mish)
adam_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

torch.manual_seed(1)

#gradient clip, policy loss
def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantage
    
    m = torch.min(ratio*advantage, clipped)
    return -m


# Initialize dictionaries to store counts of each action at each time step
action_counts = {
    "Sit": 0,
    "Lockdown/Quarantine": 0,
    "Lockdown + Cure": 0
}
    
# Training loop
episode_rewards = []
gamma = 0.98
eps = 0.2
w = tensorboard.SummaryWriter()
s = 0
max_grad_norm = 0.5

for i in range(episode_count):
    state = getState(data, 0, window_size + 1)
    prev_prob_act = None
    done = False
    total_reward = 0
        
    while not done:
        s += 1
        probs = actor(t(state))
        dist = torch.distributions.Categorical(probs=probs)
        print("Episode " + str(i) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1) #array
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
                total_cases = data[k] - recovered # at time t
                reward = max(recovered, 0)
                print("Decrease due to lockdown: " + formatCases(total_cases) + " | Cases Reduced: " + formatCases(recovered))
            elif action == 2 and len(totalinventory) > 0 and data[k]>100:  # 100% effective cure given to 10% of infected + lockdown
                recovered_cured = 0
                cured  = 0
                action_counts["Lockdown + Cure"] += 1
                if k >= recovery_period:
                    recovered_cured = oldest_cases + 0.1 * data[k]
                    cases = data[k] - recovered_cured # at time t
                    total_cases = max(cases, 0)
                    reward = max(recovered_cured, 0)
                    print("Decrease due to lockdown and cure: " + formatCases(total_cases) + " | Cases Reduced: " + formatCases(recovered_cured))
                else:
                    cured = 0.1 * data[k]
                    total_cases = data[k] - cured
                    reward = max(cured, 0)
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
        next_state = next_state.reshape(1, -1)
        
        advantage = reward + (1-done)*gamma*critic(t(next_state)) - critic(t(state))
        
        w.add_scalar("loss/advantage", advantage, global_step=s)
        w.add_scalar("actions/action_0_prob", dist.probs[0], global_step=s)
        w.add_scalar("actions/action_1_prob", dist.probs[1], global_step=s)
        
        total_reward += reward
        state = next_state
        
        if prev_prob_act:
            actor_loss = policy_loss(prev_prob_act.detach(), prob_act, advantage.detach(), eps)
            w.add_scalar("loss/actor_loss", actor_loss, global_step=s)
            adam_actor.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(adam_actor, max_grad_norm)
            w.add_histogram("gradients/actor",
                             torch.cat([p.grad.view(-1) for p in actor.parameters()]), global_step=s)
            adam_actor.step()

            critic_loss = advantage.pow(2).mean()
            w.add_scalar("loss/critic_loss", critic_loss, global_step=s)
            adam_critic.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(adam_critic, max_grad_norm)
            w.add_histogram("gradients/critic",
                             torch.cat([p.data.view(-1) for p in critic.parameters()]), global_step=s)
            adam_critic.step()
        
        prev_prob_act = prob_act
    
    w.add_scalar("reward/episode_reward", total_reward, global_step=i)
    episode_rewards.append(total_reward)
    print(episode_rewards)
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards per Episode')
    plt.savefig('pporep.png')
    
    # Plot the frequency of each action over time step 'n'
    actions = list(action_counts.keys())
    frequencies = list(action_counts.values())
    plt.figure(figsize=(6, 8))
    plt.bar(actions, frequencies)
    plt.xlabel('Actions')
    plt.ylabel('Frequency')
    plt.title('Frequency of Each Action over Time Step n')
    plt.savefig('ppofrequencies.png')

    print(f"Episode {i + 1}, Reward: {total_reward}")
