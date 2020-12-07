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

sys.path.append(".")

l1 = 78 # number of features as defined by question 1. Returned from getAgentBeliefState()
l2 = 150
l3 = 100
l4 = 6 # number of actions available -- all randomized within DeepQAgent Class

gamma = 0.85
epsilon = 0.3
epochs = 15000
losses = []
max_moves = 250
mem_size = 10000000
batch_size = 128
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
replay = deque(maxlen=mem_size)
sync_freq = 500
j=0

def get_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3,l4)
    )

    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())

    return model, model2

def run(env, agent, percept, action):
    agent, action = agent.nextAction(percept, action)
    env, percept = env.applyAction(action)

    return env, agent, percept

def getState(belief_state):
    state_ = np.array([belief_state]).reshape(1,l1) + np.random.rand(1,l1)/100.0   
    state = torch.from_numpy(state_).float()

    return state_, state

def getMoveCount(moves):
  sequence = 1
  reversed_moves = moves[::-1]        
  for index, move in enumerate(reversed_moves):
      if index + 1 == len(reversed_moves):
          break
      if sequence == 10:
          break
      if(move == reversed_moves[index + 1]):
          sequence += 1
      else:
          break 
  return sequence               

model, model2 = get_model()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
won_counter = 0
move_tracker = []

for i in range(epochs):
    terminated = False
    reward = 0
    status = 1
    mov = 0
    random_move_counter = 0
    world = WumpusWorldEnvironment()
    initialEnv, initialPercept = world.apply(4, 4, 0.2, False)        
    agent = DeepQAgent(4, 4)    
    randGen = randrange(6)
    env, agent, percept = run(initialEnv, agent, initialPercept, randGen)
    #method to return all belief state variables found in question 1 of the assignment description
    belief_state = agent.getAgentBeliefState()
    state1_, state1 = getState(belief_state)
    move_tracker.append(randGen)
    while terminated == False:
        j+=1
        mov += 1      
        qval = model(state1)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)
        nextMove = action_.item()
        move_tracker.append(nextMove)
        sameMoveCount = getMoveCount(move_tracker)

        # if (sameMoveCount >= 10):
        #     random_move_counter += 1
        #     nextMove = np.random.randint(0,6)
        
        if (random.random() < epsilon):
            random_move_counter += 1
            nextMove = np.random.randint(0,6)


        env, agent, percept = run(env, agent, percept, nextMove) 
 
        #method to return all belief state variables found in question 1 of the assignment description
        belief_state = agent.getAgentBeliefState()      
        state2_, state2 = getState(belief_state)
        reward += percept.reward
        # print("REWARD", reward)
        # print("----")
        # print("BELIEF_STATE", belief_state)
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
                Q2 = model2(state2_batch)
            
            Y = reward_batch + gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            print("Epoch =>", i) 
            print("# of wins", won_counter)
            print("# of NON random moves take per game", mov - random_move_counter)
            print("# of random moves take per game", random_move_counter)
            print(env.visualize())
            clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            if j % sync_freq == 0: 
                model2.load_state_dict(model.state_dict())  

        if reward > 0:
          won_counter += 1

        if mov > max_moves:
            terminated = True

        if percept.isTerminated == True:
            terminated = True

losses = np.array(losses)            
