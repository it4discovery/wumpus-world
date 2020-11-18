import sys
sys.path.append(".")
from random import randrange
from wumpusWorld.environment.Environment import Action, Percept
from wumpusWorld.environment.Agent import Agent

class NaiveAgent():

    def nextAction(self, percept):
        randGen = randrange(6)
        if randGen == 0:
            return Action.Forward
        elif randGen == 1:
            return Action.TurnLeft
        elif randGen == 2:
            return Action.TurnRight
        elif randGen == 3:
            return Action.Shoot
        elif randGen == 4:
            return Action.Grab
        elif randGen == 5:
            return Action.Climb
