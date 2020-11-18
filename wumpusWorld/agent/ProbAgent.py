from pomegranate import *
from wumpusWorld.environment.Agent import Agent
from wumpusWorld.environment.Environment import Coords, Action, Percept
from wumpusWorld.environment.Orientation import East, North, South, West
from networkx import nx
from copy import deepcopy
from random import randrange
import sys
sys.path.append(".")


class ProbAgent():
    def __init__(self, gridHeight=4, gridWidth=4, pitProb=0.2, agentState=Agent, safeLocations=[Coords(0,0)], beelineActionList=[], nextBestCoords=[], stenchLocations=[], breezeLocations=[], heardScream=False, lastLocation = Coords(0,0), counter=0):
        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.agentState = agentState()
        self.safeLocations = safeLocations
        self.beelineActionList = beelineActionList
        self.nextBestCoords = nextBestCoords
        self.counter = counter
        self.heardScream = heardScream
        self.stenchLocations = stenchLocations
        self.breezeLocations = breezeLocations
        self.lastLocation = lastLocation
        
    def buildModel(self, neighbours, dist, observe=True):
        prob = 1
        newNeighbour = []
        for item in neighbours:            
            if any(d.y != item.y and d.x != item.x for d in self.safeLocations):  
                newNeighbour.append(item)
        if(len(newNeighbour) > 2):
            neighbours = newNeighbour
     
        if(len(neighbours) == 2):
            n1 = DiscreteDistribution(dist)
            n2 = DiscreteDistribution(dist)
            ct = ConditionalProbabilityTable(
                [
                    [True,  True,  True,  1.0],
                    [True,  True,  False, 0.0],
                    [True,  False, True,  1.0],
                    [True,  False, False, 0.0],
                    [False, True,  True,  1.0],
                    [False, True,  False, 0.0],
                    [False, False, True,  0.0],
                    [False, False, False, 1.0]],
                [n1, n2],
            )
            s1 = Node(n1, name="n1")
            s2 = Node(n2, name="n2")
            s3 = Node(ct, name="ct")
            model = BayesianNetwork("two-neighbour")
            model.add_nodes(s1, s2, s3)
            model.add_edge(s1, s3)
            model.add_edge(s2, s3)
            model.bake()
            beliefs = model.predict_proba({"ct": observe})
            
            for x in beliefs[1].parameters[0]:
                if(x == observe):
                    prob = round(beliefs[1].parameters[0][x], 3)
    

        elif(len(neighbours) == 3):
            n1 = DiscreteDistribution(dist)
            n2 = DiscreteDistribution(dist)
            n3 = DiscreteDistribution(dist)
            ct = ConditionalProbabilityTable(
                [
                    [True, True, True, True, 1.0],
                    [True, True, True, False, 0.0],
                    [True, True, False, True, 1.0],
                    [True, True, False, False, 0.0],
                    [True, False, True, True, 1.0],
                    [True, False, True, False, 0.0],
                    [True, False, False, True, 1.0],
                    [True, False, False, False, 0.0],
                    [False, True, True, True, 1.0],
                    [False, True, True, False, 0.0],
                    [False, True, False, True, 1.0],
                    [False, True, False, False, 0.0],
                    [False, False, True, True, 1.0],
                    [False, False, True, False, 0.0],
                    [False, False, False, True, 0.0],
                    [False, False, False, False, 1.0]
                ],
                [n1, n2, n3])

            s1 = Node(n1, name="n1")
            s2 = Node(n2, name="n2")
            s3 = Node(n3, name="n3")
            s4 = Node(ct, name="ct")

            model = BayesianNetwork("three-neighbour")
            model.add_nodes(s1, s2, s3, s4)
            model.add_edge(s1, s4)
            model.add_edge(s2, s4)
            model.add_edge(s3, s4)
            model.bake()
            beliefs = model.predict_proba({"ct": observe})

            for x in beliefs[2].parameters[0]:
                if(x == observe):
                    prob = round(beliefs[2].parameters[0][x], 3)

        elif(len(neighbours) == 4):
            n1 = DiscreteDistribution(dist)
            n2 = DiscreteDistribution(dist)
            n3 = DiscreteDistribution(dist)
            n4 = DiscreteDistribution(dist)
            ct = ConditionalProbabilityTable(
                [
                    [True, True, True, True, True, 1.0],
                    [True, True, True, True, False, 0.0],
                    [True, True, True, False, True, 1.0],
                    [True, True, True, False, False, 0.0],
                    [True, True, False, True, True, 1.0],
                    [True, True, False, True, False, 0.0],
                    [True, True, False, False, True, 1.0],
                    [True, True, False, False, False, 0.0],
                    [True, False, True, True, True, 1.0],
                    [True, False, True, True, False, 0.0],
                    [True, False, True, False, True, 1.0],
                    [True, False, True, False, False, 0.0],
                    [True, False, False, True, True, 1.0],
                    [True, False, False, True, False, 0.0],
                    [True, False, False, False, True, 1.0],
                    [True, False, False, False, False, 0.0],
                    [False, True, True, True, True, 1.0],
                    [False, True, True, True, False, 0.0],
                    [False, True, True, False, True, 1.0],
                    [False, True, True, False, False, 0.0],
                    [False, True, False, True, True, 1.0],
                    [False, True, False, True, False, 0.0],
                    [False, True, False, False, True, 1.0],
                    [False, True, False, False, False, 0.0],
                    [False, False, True, True, True, 1.0],
                    [False, False, True, True, False, 0.0],
                    [False, False, True, False, True, 1.0],
                    [False, False, True, False, False, 0.0],
                    [False, False, False, True, True, 1.0],
                    [False, False, False, True, False, 0.0],
                    [False, False, False, False, True, 0.0],
                    [False, False, False, False, False, 1.0]
                ],
                [n1, n2, n3, n4])

            s1 = Node(n1, name="n1")
            s2 = Node(n2, name="n2")
            s3 = Node(n3, name="n3")
            s4 = Node(n4, name="n4")
            s5 = Node(ct, name="ct")

            model = BayesianNetwork("four-neighbour")
            model.add_nodes(s1, s2, s3, s4, s5)
            model.add_edge(s1, s5)
            model.add_edge(s2, s5)
            model.add_edge(s3, s5)
            model.add_edge(s4, s5)
            model.bake()
            beliefs = model.predict_proba({'ct': observe})

            for x in beliefs[3].parameters[0]:
                if(x == observe):
                    prob = round(beliefs[3].parameters[0][x], 3)

        return prob

    def buildEscapeRoute(self, safeLocations):
        G = nx.grid_2d_graph(self.gridWidth, self.gridHeight)

        for location in safeLocations:
            nx.add_path(G, (location.x, location.y))

        escapeRoute = list(nx.shortest_path(
            G, (safeLocations[-1].x, safeLocations[-1].y), (0, 0)))

        return escapeRoute

    def shouldBeFacing(self, currentItem, nextItem):
        currX = currentItem[0]
        currY = currentItem[1]
        nextX = nextItem[0]
        nextY = nextItem[1]

        if(nextY < currY):
            return South
        elif(nextX < currX):
            return West

    def getOrientation(self, agentLocation, dest):
        if(agentLocation is None or dest is None):
            sys.exit()

        if(dest.y < agentLocation.y):
            return South
        elif(dest.y > agentLocation.y):
            return North
        elif(dest.x < agentLocation.x):
            return West
        elif(dest.x > agentLocation.x):
            return East

    def determineNextAction(self, agentOrientation, escapeRouteActions):
        currentEscapeAction = Action.Forward

        if(agentOrientation != self.shouldBeFacing(escapeRouteActions[0], escapeRouteActions[1])):
            currentEscapeAction = Action.TurnRight
        else:
            escapeRouteActions = escapeRouteActions[1:]

        return currentEscapeAction, escapeRouteActions

    def neighbours(self):
        rows = []
        for i in range(self.gridHeight):
            cols = []
            for j in range(self.gridWidth):
                coords = Coords(i, j)
                arr = coords.adjacentCells(self.gridWidth, self.gridHeight)
                cols.append(list(filter(None, arr)))

            rows.append(cols)

        return rows

    def getNextBestCoords(self, currentCoords, pitProbs, wumpusProbs):
        bestProb = 1
        bestCoords = Coords(0,0)
        availableCells = list(
            filter(None, currentCoords.adjacentCells(self.gridWidth, self.gridHeight)))
        lastVisited = self.lastLocation
        probabilityList = pitProbs 
        
        for item in availableCells:            
            prob = pitProbs[item.y][item.x] + wumpusProbs[item.y][item.x]

            if(prob < 0.25 and prob > 0.0):
                bestCoords = item 
                break
            elif(prob <= 0.5 and prob > 0.0):
                bestCoords = item 
                break
            elif(prob > 0.5 and prob > 0.0):
                #not many good options. Take a guess
                randGen = randrange(len(availableCells))
                bestCoords = availableCells[randGen]
                break
            else:
                #no good moves, heading out
                escapeCoords = self.buildEscapeRoute(self.safeLocations)
                escapeCoords = escapeCoords[0]               
                bestCoords = Coords(0, 0)
               
                
        return bestCoords

    def pitProb(self, breezeLocations, safeLocations):
        neighbours = self.neighbours()

        rows = []
        for i in range(self.gridHeight):
            cols = []
            for j in range(self.gridWidth):
                prob = 0.2
                cols.append(prob)

            rows.append(cols)

        for i in range(self.gridHeight):
            for j in range(self.gridWidth):            
                if any(d.y == i and d.x == j for d in breezeLocations):                        
                    prob = self.buildModel(
                        neighbours[i][j], {True: 0.20, False: 0.80}, True)
                    
                    for item in neighbours[i][j]:
                        rows[item.x][item.y] = prob

        for i in range(self.gridHeight):
            for j in range(self.gridWidth):   
                if any(d.y == i and d.x == j for d in safeLocations):
                    rows[i][j] = 0.0 

        rows[0][0] = 0.0

        return rows

    def wumpusProb(self, stenchLocations, safeLocations):
        neighbours = self.neighbours()
    
        rows = []

        for i in range(self.gridHeight):
            cols = []
            for j in range(self.gridWidth):
                prob = round(1./15, 3)
                cols.append(prob)

            rows.append(cols)

        for i in range(self.gridHeight):
            for j in range(self.gridWidth):            
                if any(d.y == i and d.x == j for d in stenchLocations):                        
                    prob = self.buildModel(
                        neighbours[i][j], {True: 1./15, False: 14./15}, True)
                    
                    for item in neighbours[i][j]:
                        rows[item.x][item.y] = prob

        for i in range(self.gridHeight):
            for j in range(self.gridWidth):   
                if any(d.y == i and d.x == j for d in safeLocations):
                    rows[i][j] = 0.0 

        rows[0][0] = 0.0

        return rows

    def printProbTable(self, grid):
        rows = []
        for i in range(self.gridHeight):
            cells = []
            for j in range(self.gridWidth):            
                cells.append("%s" % (grid[i][j]))

            rows.append('|'.join(cells))
        return '\n'.join(rows)

    def nextAction(self, percept):
        ret = deepcopy(self)

        if(percept.stench == True):
            ret.stenchLocations.append(ret.agentState.location)
        if(percept.breeze == True):
            ret.breezeLocations.append(ret.agentState.location)

        pitProbabilityGrid = self.pitProb(ret.breezeLocations, ret.safeLocations)
        wumpusProbabilityGrid = self.wumpusProb(ret.stenchLocations, ret.safeLocations) if self.heardScream is False else []
                      
        nextBestCoords = self.getNextBestCoords(ret.agentState.location, pitProbabilityGrid, wumpusProbabilityGrid)

        print("PIT PROB TABLE")
        print(self.printProbTable(pitProbabilityGrid)) 
        print("-- --")
        print("WUMPUS PROB TABLE")
        print(self.printProbTable(wumpusProbabilityGrid))         
        ret.counter = ret.counter + 1
        #sys.exit()
        if ret.counter == 100:
            sys.exit()  # if it's taken 1000 moves, let's bail

        if ret.agentState.hasGold == True:
            if(self.agentState.location.x == Coords(0, 0).x and self.agentState.location.y == Coords(0, 0).y):
                return ret, Action.Climb
            else:
                escapeRouteActions = self.buildEscapeRoute(self.safeLocations) if len(
                    self.beelineActionList) == 0 else self.beelineActionList

                currentEscapeAction, escapeRouteActions = self.determineNextAction(
                    ret.agentState.orientation, escapeRouteActions)

                ret.agentState = ret.agentState.applyMoveAction(
                    currentEscapeAction, self.gridWidth, self.gridHeight)
                ret.beelineActionList = escapeRouteActions

                return ret, currentEscapeAction   
        elif percept.scream == True:
            ret.agentState = ret.agentState.useArrow()
            ret.heardScream = True
            return ret, Action.Shoot        
        elif percept.glitter == True:
            ret.agentState.hasGold = True
            return ret, Action.Grab
        else:
            orientation = ret.agentState.orientation
            location = ret.agentState.location
            nextOrientation = self.getOrientation(location, nextBestCoords)

            if(nextOrientation == orientation):
                ret.agentState = ret.agentState.forward(
                    self.gridWidth, self.gridHeight)

                ret.lastLocation = ret.agentState.location
                ret.safeLocations.append(ret.agentState.location)
                ret.nextBestCoords = []
                return ret, Action.Forward
            else:
                ret.agentState = ret.agentState.turnRight()
                return ret, Action.TurnRight
