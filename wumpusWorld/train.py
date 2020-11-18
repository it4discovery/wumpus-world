#Install PyTorch
#pip3 install numpy
#pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

#Install IPython
#pip3 install IPython

#Install matplotlib
#pip3 install matplotlib

import sys
import copy
import numpy as np
import torch
from IPython.display import clear_output
import random
from matplotlib import pylab as plt
from collections import deque

sys.path.append(".")

from environment.Environment import Action, WumpusWorldEnvironment, Percept
from agent.DeepQAgent import DeepQAgent

l1 = 72 # number of features as defined by question 1. Returned from getAgentBeliefState()
l2 = 150
l3 = 100
l4 = 6 # number of actions available -- all randomized within DeepQAgent Class

gamma = 0.9
epsilon = 0.3
epochs = 10

def run(env, agent, percept):
    agent, action = agent.nextAction(percept)
    env, percept = env.applyAction(action)

    return env, agent, percept

for i in range(epochs):
    terminated = False
    reward = 0
    world = WumpusWorldEnvironment()
    initialEnv, initialPercept = world.apply(4, 4, 0.2, False)        
    agent = DeepQAgent(4, 4)    
    env, agent, percept = run(initialEnv, agent, initialPercept)

    while terminated == False:
        env, agent, percept = run(env, agent, percept)
        reward += percept.reward
        
        #method to return all belief state variables found in question 1 of the assignment description
        agentLocationGrid, safeLocationGrid, stenchLocationGrid, breezeLocationGrid, agentHasGold, agentSensesGold, agentHasArrow, agentHeardScream, agentOrientationSet = agent.getAgentBeliefState()

        #helper method to print the agent's belief state
        #feel free to comment this call out
        agent.printBeliefState(percept)
        print(percept.show())
        print(env.visualize())
        
        if percept.isTerminated == True:
            terminated = True