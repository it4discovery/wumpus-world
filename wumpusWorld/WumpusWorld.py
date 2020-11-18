import sys
sys.path.append(".")

from environment.Environment import Action, WumpusWorldEnvironment, Percept
# from agent.NaiveAgent import NaiveAgent
# from agent.BeelineAgent import BeelineAgent
# from agent.ProbAgent import ProbAgent
from agent.DeepQAgent import DeepQAgent


class WumpusWorld:
    def __init__(self):
        self.reward = 0

        world = WumpusWorldEnvironment()
        initialEnv, initialPercept = world.apply(4, 4, 0.2, False)        
        agent = DeepQAgent(4, 4)
        self.runEpisode(initialEnv, agent, initialPercept)
        print("Total reward: %s" % self.reward)

    def runEpisode(self, env, agent, percept):
        newAgent, nextAction = agent.nextAction(percept)

        nextEnvironment, nextPercept = env.applyAction(nextAction)
        self.reward += nextPercept.reward
        print(nextPercept.show())
        print(nextEnvironment.visualize())

        if nextPercept.isTerminated == False:                        
            self.runEpisode(nextEnvironment, newAgent, nextPercept)


world = WumpusWorld()
