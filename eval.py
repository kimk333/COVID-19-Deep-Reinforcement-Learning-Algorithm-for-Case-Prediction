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

file_name = input("Enter file_name, Model_name")
model_name = input()
model = load_model(model_name) # input the folder name
window_size = model.layers[0].input.shape.as_list()[1]
agent = DQNAgent(window_size, True, model_name)
data = getCaseDataVec(file_name)
print(data)
recovery_period = 14
l = len(data) - 1
batch_size = 32

action_counts = {
    "Sit": 0,
    "Lockdown/Quarantine": 0,
    "Lockdown + Cure": 0
}
# Initialize a list to store cumulative reduction in cases over episodes
cumulative_reduction = []
state = getState(data, 0, window_size + 1)
print(state)
total_cases = 0
total_reward = 0
cases_reduced = 0 
agent.inventory = []
print(l)
for t in range(l):
    action = agent.act(state)
    next_state = getState(data, t + 1, window_size + 1)
    reward = 0
    if action == 0: # sit
        agent.inventory.append(data[t])
        total_cases = data[t]
        action_counts["Sit"] += 1
        reward = 0
        print("No control: " + formatCases(data[t]))
    elif action == 1 and len(agent.inventory) > 0: # lockdown/quarantine
        oldest_cases = agent.inventory.pop(0)
        recovered = 0
        if t >= recovery_period:
            recovered = oldest_cases
            action_counts["Lockdown/Quarantine"] += 1
        reward = max(recovered, 0)
        total_reward += reward
        total_cases = data[t] - recovered # at time t
        cases_reduced += recovered
        print("Decrease due to lockdown: " + formatCases(total_cases) + " | Cases Reduced: " + formatCases(recovered))
    elif action == 2 and len(agent.inventory) > 0 and data[t]>100:  # 100% effective cure given to 10% of infected + lockdown
        recovered_cured = 0
        cured  = 0
        action_counts["Lockdown + Cure"] += 1
        if t >= recovery_period:
            recovered_cured = oldest_cases + 0.1 * data[t]
            reward = max(recovered_cured, 0)
            total_reward += reward
            cases = data[t] - recovered_cured # at time t
            total_cases = max(cases, 0)
            cases_reduced += recovered_cured
            print("Decrease due to lockdown and cure: " + formatCases(total_cases) + " | Cases Reduced: " + formatCases(recovered_cured))
        else:
            cured = 0.1 * data[t]
            reward = max(cured, 0)
            total_reward += reward
            total_cases = data[t] - cured
            cases_reduced += cured
            print("Decrease due to cure: " + formatCases(total_cases) + " | Cases Reduced: " + formatCases(cured))
    # Store the total cases reduced in the current episode in the cumulative_reduction list
    cumulative_reduction.append(cases_reduced)
    done = True if t == l - 1 else False
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state
    if done:
        print("--------------------------------")
        print(file_name + " Total Cases: " + formatCases(total_cases))
        print("--------------------------------")
        print("The total amount of cases is:",formatCases(total_cases))
    # Plot the frequency of each action over time step 'n'
    actions = list(action_counts.keys())
    frequencies = list(action_counts.values())
    plt.figure(figsize=(8, 6))
    plt.bar(actions, frequencies)
    plt.xlabel('Actions')
    plt.ylabel('Frequency')
    plt.title('Frequency of Each Action over Time Step n')
    plt.savefig('frequencies.png')

    # Plot the amount of cases reduced over episodes as a line plot
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_reduction)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Cases Reduced')
    plt.title('Amount of Cases Reduced over Episodes')
    plt.savefig('recovered.png')
    
    # Plot the total cases at the end of each episode as a line plot
    plt.figure(figsize=(8, 6))
    plt.plot(total_cases_end_of_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Total Cases')
    plt.title('Total Cases over Episodes')
    plt.savefig('total.png')

