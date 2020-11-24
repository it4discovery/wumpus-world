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
from random import randrange

sys.path.append(".")

from environment.Environment import Action, WumpusWorldEnvironment, Percept
from agent.DeepQAgent import DeepQAgent

l1 = 80 # number of features as defined by question 1. Returned from getAgentBeliefState()
l2 = 150
l3 = 100
l4 = 6 # number of actions available -- all randomized within DeepQAgent Class

gamma = 0.9
epsilon = 0.3
epochs = 1

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
)

model2 = copy.deepcopy(model) #A
model2.load_state_dict(model.state_dict()) #B

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def run(env, agent, percept, action):
    agent, action = agent.nextAction(percept, action)
    env, percept = env.applyAction(action)

    return env, agent, percept

def getState(agentLocationGrid, safeLocationGrid, stenchLocationGrid, breezeLocationGrid, agentHasGold, agentSensesGold, agentHasArrow, agentHeardScream, agentOrientationSet):
    state_ = np.array([agentLocationGrid, safeLocationGrid, stenchLocationGrid, breezeLocationGrid, agentOrientationSet]).reshape(1,l1) + np.random.rand(1,l1)/10.0
    state = torch.from_numpy(state_).float()

    return state_, state

for i in range(epochs):
    terminated = False
    reward = 0
    world = WumpusWorldEnvironment()
    initialEnv, initialPercept = world.apply(4, 4, 0.2, False)        
    agent = DeepQAgent(4, 4)    
    randGen = randrange(6)
    env, agent, percept = run(initialEnv, agent, initialPercept, randGen)
    #method to return all belief state variables found in question 1 of the assignment description
    agentLocationGrid, safeLocationGrid, stenchLocationGrid, breezeLocationGrid, agentHasGold, agentSensesGold, agentHasArrow, agentHeardScream, agentOrientationSet = agent.getAgentBeliefState()
    
    state_, state = getState(agentLocationGrid, safeLocationGrid, stenchLocationGrid, breezeLocationGrid, agentHasGold, agentSensesGold, agentHasArrow, agentHeardScream, agentOrientationSet)
    
    counter = 0
    nonForwardCounter = 0

    while terminated == False:
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)

        nextMove = action_.item()

        env, agent, percept = run(env, agent, percept, nextMove) 
 
        #method to return all belief state variables found in question 1 of the assignment description
        agentLocationGrid, safeLocationGrid, stenchLocationGrid, breezeLocationGrid, agentHasGold, agentSensesGold, agentHasArrow, agentHeardScream, agentOrientationSet = agent.getAgentBeliefState()
        state_, state = getState(agentLocationGrid, safeLocationGrid, stenchLocationGrid, breezeLocationGrid, agentHasGold, agentSensesGold, agentHasArrow, agentHeardScream, agentOrientationSet)
        reward += percept.reward
        #helper method to print the agent's belief state
        #feel free to comment this call out
        #agent.printBeliefState(percept)
        #print(percept.show())
        
        print(env.visualize())
        # if counter == 50 :
        #     sys.exit()
        counter = counter + 1
        if percept.isTerminated == True:
            terminated = True