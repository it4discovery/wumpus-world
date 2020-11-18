
import sys
sys.path.append(".")

from wumpusWorld.environment.Environment import Action, Percept
from wumpusWorld.agent.NaiveAgent import NaiveAgent

class LogicalAgent:
    def __init__(self):
        self.dumbDumb = NaiveAgent()

    def nextAction(self, percept):
        return Action.Forward
