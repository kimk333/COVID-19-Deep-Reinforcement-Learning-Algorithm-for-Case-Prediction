import tensorflow
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from DQN import DQNAgent
from func import *
import math
import numpy as np
import random
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
from collections import deque

import sys
import matplotlib.pyplot as plt
import numpy as np

stock_name = input("Enter stock_name: ") # input .csv file name
window_size = int(input("Enter window_size: ")) # input time period, integer
episode_count = int(input("Enter Episode_count: ")) # input number of episodes to train

stock_name = str(stock_name)
window_size = int(window_size)
episode_count = int(episode_count)
agent = DQNAgent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

episode_rewards = []

for e in range(episode_count + 1):
    total_reward = 0
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    for t in range(l):
        action = agent.act(state)
        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        if action == 1: # buy
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = window_size_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
        total_reward += reward
        episode_rewards.append(total_reward)
        print(episode_rewards)

        # Plot rewards per episode after each episode
        plt.figure(figsize=(8, 6))
        plt.plot(episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Rewards per Episode')
        plt.savefig('dqnrep.png')
