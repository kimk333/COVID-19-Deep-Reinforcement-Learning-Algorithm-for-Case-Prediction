import tensorflow
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import math
import numpy as np
import random
import csv
import pandas as pd
import sys
from collections import deque

# Define the function to format the case numbers
def formatCases(n):
    return "{0:,}".format(int(n))

# Define the function to get the COVID-19 case data
def getCaseDataVec(key):
    vec = []
    lines = open(key+".csv","r").read().splitlines()
    for line in lines[1:]:
        vec.append(float(line.split(",")[2]))
    return vec

# Define the sigmoid function for calculation
def sigmoid(x):
    return 1/(1+math.exp(-x))

# Returns the current state of the data
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])
