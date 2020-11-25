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

l1 = 72 # number of features as defined by question 1. Returned from getAgentBeliefState()
l2 = 150
l3 = 100
l4 = 6 # number of actions available -- all randomized within DeepQAgent Class

gamma = 0.9
epsilon = 0.3
epochs = 1000
losses = []
max_moves = 50
mem_size = 1000
batch_size = 200
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
replay = deque(maxlen=mem_size)
sync_freq = 500 #A
j=0

def get_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3,l4)
    )

    model2 = copy.deepcopy(model) #A
    model2.load_state_dict(model.state_dict()) #B   

    return model, model2

def run(env, agent, percept, action):
    agent, action = agent.nextAction(percept, action)
    env, percept = env.applyAction(action)

    return env, agent, percept

def getState(belief_state):
    state_ = np.array([belief_state]).reshape(1,l1) + np.random.rand(1,l1)/100.0   
    state = torch.from_numpy(state_).float()

    return state_, state

model, model2 = get_model()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(epochs):
    terminated = False
    reward = 0
    status = 1
    mov = 0
    world = WumpusWorldEnvironment()
    initialEnv, initialPercept = world.apply(4, 4, 0.2, False)        
    agent = DeepQAgent(4, 4)    
    randGen = randrange(6)
    env, agent, percept = run(initialEnv, agent, initialPercept, randGen)
    #method to return all belief state variables found in question 1 of the assignment description
    belief_state = agent.getAgentBeliefState()
    state1_, state1 = getState(belief_state)

    while terminated == False:
        qval = model(state1)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)

        nextMove = action_.item()
        if (random.random() < epsilon):
            nextMove = np.random.randint(0,6)
        else:
            nextMove = np.argmax(qval_)

        env, agent, percept = run(env, agent, percept, nextMove) 
 
        #method to return all belief state variables found in question 1 of the assignment description
        belief_state = agent.getAgentBeliefState()
        state2_, state2 = getState(belief_state)
        reward += percept.reward

        exp =  (state1, action_, reward, state2, percept.isTerminated)
        replay.append(exp)
        state1 = state2     

        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size) 
            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])

            Q1 = model(state1_batch) 
            with torch.no_grad():
                Q2 = model2(state2_batch) #B
            
            Y = reward_batch + gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            print(i, loss.item()) 
            clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            if j % sync_freq == 0: 
                model2.load_state_dict(model.state_dict())  

        if reward != -1 or mov > max_moves:
            status = 0
            mov = 0

        if percept.isTerminated == True:
            terminated = True

losses = np.array(losses)            