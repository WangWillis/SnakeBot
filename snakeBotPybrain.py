# Snake code by sanchitgangwar
# Genetic Learning Alg code by Willis Wang
# SNAKES GAME
# Use ARROW KEYS to play, SPACE BAR for pausing/resuming and Esc Key for exiting

import copy
import math
import numpy as np
from numpy import exp, array, random, dot
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint

def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetwork():
    def __init__(self, inp, hidden, output):
        #initialize sizes
        self.inp = inp+1
        self.hidden = hidden
        #self.mHidden = hidden/4
        self.output = output
    
        #initialize the arrays for the layers
        self.il = [1.0] * self.inp
        self.hl = [1.0] * self.hidden
        #self.hhl = [1.0] * self.mHidden
        self.ol = [1.0] * self.output

        #randomize weights
        self.wi = np.random.randn(self.inp, self.hidden)
        #self.wh = np.random.randn(self.hidden, self.mHidden)
        self.wo = np.random.randn(self.hidden, self.output)

    def __getitem__(self):
        return self

    def think(self, inp):
        if len(inp) != self.inp-1:
            return ValueError("BAD INPUT")
        softMax = 0
        #copy data to input
        for i in range(self.inp-1):
            self.il[i] = inp[i]
    
        #propigate through hidden layers
        for j in range(self.hidden):
            tot = 0.0 
            #loop through hidden layer to change values
            for i in range(self.inp):
                tot += self.il[i]*self.wi[i][j]
            #set hidden layer values
            self.hl[j] = sigmoid(tot)

        # #propigate through hidden layers
        # for j in range(self.mHidden):
        #     tot = 0.0 
        #     #loop through hidden layer to change values
        #     for i in range(self.hidden):
        #         tot += self.hl[i]*self.wh[i][j]
        #     #set hidden layer values
        #     self.hhl[j] = sigmoid(tot)

        #propigate activations
        for j in range(self.output):
            tot = 0.0 
            #loop through hidden layer to change values
            for i in range(self.hidden):
                tot += self.hl[i]*self.wo[i][j]
            #set hidden layer values
            sig = sigmoid(tot)
            self.ol[j] = sig
            softMax += exp(sig)

        #calculate prob using softMax
        for i in range(self.output):
            self.ol[i] = exp(self.ol[i])/softMax

        return self.ol

screenX = 20
screenY = 60
inputSize =7
numHiddenLayers = 300
numOutputs = 4
numChrome = 9
gen = 0
key = 0
KEYS = [KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN, KEY_RIGHT]

fitnesses = [0] * numChrome
population = [0] * numChrome
global bestNN
bestNN = 0
global bestFit
bestFit = 1
global bestScore
bestScore = -1
global bestGen 
bestGen = 0

def createScreen(snake, food):
    screen = [0]*inputSize

    for i in range(0, len(snake)):
        screen[snake[i][0]*screenY+snake[i][1]] = 5*(snake[i][0]+snake[i][1])
    screen[snake[0][0]*screenY+snake[0][1]] = 20*(snake[0][0]+snake[0][1])
    screen[food[0]*screenY+food[1]] = 50*(snake[0][0]+snake[0][1])
 
    
    return screen

def getSnakeVal(snake):
    badTouch = 0
    for i in range(1, len(snake)):
        badTouch += i*(snake[0][0]-snake[i][0])
        badTouch += (len(snake)-i)*(snake[0][1]-snake[i][1])
    return badTouch

def getBiggestIndex(arr):
    index = 0
    for i in range(1, len(arr)):
        if arr[i] > arr[index]:
            index = i
    return index

def fitness(res):
    return 1/(((exp(res[0])-1)/(5))+((math.log(res[1]*100+1)/100))+1)#+(1/(res[2]-res[0]+1))+1)

def getBadDis(snake, grid, key):
    cullX = [snake[0][0]] * 3
    cullY = [snake[0][1]] * 3
    colPosX = 0
    colPosY = 0
    if key == KEY_LEFT:
        colPosY = -1
    elif key == KEY_RIGHT:
        colPosY = 1
    elif key == KEY_UP:
        colPosX = -1
    else:
        colPosX = 1

    while cullX[0] > 0 and cullX[0] < screenX or cullY[0] > 0 and cullY[0] < screenY:
        if grid[cullX[0]][cullY[0]] == 1:
            break
        cullX[0] += (-1)*colPosY
        cullY[0] += colPosX
    
    while cullX[1] > 0 and cullX[1] < screenX or cullY[1] > 0 and cullY[1] < screenY:
        if grid[cullX[1]][cullY[1]] == 1:
            break
        cullX[1] += colPosX
        cullY[1] += colPosY

    while cullX[2] > 0 and cullX[2] < screenX or cullY[2] > 0 and cullY[2] < screenY:
        if grid[cullX[2]][cullY[2]] == 1:
            break
        cullX[2] += colPosY
        cullY[2] += (-1)*colPosX
    
    cullX[0] = abs(cullX[0]-snake[0][0])
    if cullX[0] == 0:
        cullX[0] = abs(cullY[0]-snake[0][1])

    cullX[1] = abs(cullX[1]-snake[0][0])
    if cullX[1] == 0:
        cullX[1] = abs(cullY[1]-snake[0][1])
    
    cullX[2] = abs(cullX[2]-snake[0][0])
    if cullX[2] == 0:
        cullX[2] = abs(cullY[2]-snake[0][1])

    return cullX

def runGame(nn):
    curses.initscr()
    curses.noecho()
    curses.curs_set(0)
    win = curses.newwin(screenX, screenY, 0, 0)
    win.keypad(1)
    win.border(0)
    win.nodelay(1)
    global key
    key = KEY_RIGHT                                                # Initializing values
    score = 0
    timeAlive = 0
    timeLastScore = 0 
    avgDist = 0
    results = []
    
    grid = [[0 for x in range(screenY)] for y in range(screenX)]
    snake = [[4,10], [4,9], [4,8]]                                     # Initial snake co-ordinates
    grid[4][10] = 1
    grid[4][9] = 1
    grid[4][8] = 1
    food = []

    leftH = 0
    rightH = 0
    frontH = 0

    while food == []:
        food = [randint(1, 18), randint(1, 58)]                 # Calculating next food's coordinates
        if food in snake: food = []
    win.addch(food[0], food[1], '*')

    while key != 27:
        timeAlive += 0.01                                            # While Esc key is not pressed
        win.border(0)
        win.addstr(0, 2, 'Score: ' + str(score) + ' ')                # Printing 'Score' and
        win.addstr(0, 15, 'Gen: ' + str(gen) + ' ')
        win.addstr(19, 30, 'Sec: ' + str(timeAlive) + ' ')
        win.addstr(0, 27, 'fitness: ' + str(fitness([score, timeAlive, avgDist/(timeAlive*100)])))                # 'SNAKE' strings
        #win.timeout(5)
        
        
        xDis = snake[0][0]-food[0]
        yDis = snake[0][1]-food[1]
        avgDist += math.sqrt(xDis*xDis+yDis*yDis)
        cull = getBadDis(snake, grid, key)
        prevKey = key
        KEYS[4] = prevKey
        event = KEYS[getBiggestIndex(nn.think([math.pow(xDis, 3), xDis, math.pow(yDis, 3), yDis, cull[0], cull[1], cull[2]]))]
        done = win.getch()
        key = done if done == 27 else event 
        
        if ((key == KEY_LEFT and prevKey == KEY_RIGHT) or (key == KEY_RIGHT and prevKey == KEY_LEFT) or (key == KEY_UP and prevKey == KEY_DOWN) or (key == KEY_DOWN and prevKey == KEY_UP)):
            key = prevKey

        snake.insert(0, [snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1), snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1)])
        
        if snake[0][0] <= 0 or snake[0][0] >= screenX-1 or snake[0][1] <= 0 or snake[0][1] >= screenY-1: break

        if snake[0] in snake[1:]: break
        
        if snake[0] == food:                                            # When snake eats the food
            food = []
            score += 1
            timeLastScore = timeAlive

            while food == []:
                food = [randint(1, 18), randint(1, 58)]                 # Calculating next food's coordinates
                if food in snake: food = []
            win.addch(food[0], food[1], '*')
        else:    
            last = snake.pop()                                          # [1] If it does not eat the food, length decreases
            win.addch(last[0], last[1], ' ')
            grid[last[0]][last[1]] = 0
        win.addch(snake[0][0], snake[0][1], '#')
        grid[snake[0][0]][snake[0][1]] = 1

        if (timeAlive-timeLastScore) > 1.5:
            break
        key = win.getch()
        
    results.append(score)
    results.append(timeAlive)
    results.append(avgDist/(timeAlive*100))
    curses.endwin()
    return results

def breed(nn, bestNN, fit):
    prob = 0.02+fit*0.78
    nnNew = copy.deepcopy(nn)

    size = nnNew.wi.shape 
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wi[i][j] = bestNN.wi[i][j]

    # size = nnNew.wh.shape 
    # for i in range(0, size[0]):
    #     for j in range(0, size[1]):
    #         if random.random() < prob:
    #             nnNew.wh[i][j] = bestNN.wh[i][j]
    
    size = nnNew.wo.shape 
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wo[i][j] = bestNN.wo[i][j]
    return nnNew

def mutate(nn, fit, avgFit):
    random.seed()
    nnNew = copy.deepcopy(nn)
    prob = 0.02+fit*0.78
    size = nnNew.wi.shape
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wi[i][j] += np.random.randn()*2*avgFit

    # size = nnNew.wh.shape
    # for i in range(0, size[0]):
    #     for j in range(0, size[1]):
    #         if random.random() < prob:
    #             nnNew.wh[i][j] += np.random.randn()*2*avgFit

    size = nnNew.wo.shape
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wo[i][j] += np.random.randn()*2*avgFit

    return nnNew

def getMostFit(fitnesses):
    random.seed()
    mostFit = [0, 0, 0.0]
    for i in range(len(fitnesses)):
        mostFit[2] += fitnesses[i]
        if fitnesses[i] < fitnesses[mostFit[0]]:
            mostFit[1] = mostFit[0]
            mostFit[0] = i
        if fitnesses[i] == fitnesses[mostFit[0]] and random.random() < 0.5:
            mostFit[1] = mostFit[0]
            mostFit[0] = i

    mostFit[2] /= len(fitnesses)
    return mostFit



#init starting off neural networks
for i in range(len(population)):
    population[i] = NeuralNetwork(inputSize, numHiddenLayers, numOutputs)

welp = 0
random.seed()
#loop for 1000 generations
while key != 27:
    gen += 1
    #play game for whole population
    for i in range(len(population)):
        fitnesses[i] = 0
        #run each nn 4 times and average its fitness
        for j in range(0,4):
            res = runGame(population[i])
            if key == 27:
                break  
            fitnesses[i] += fitness(res)
        fitnesses[i] /= 4
        if key == 27:
            break
        #print data
        print "Gen: " + str(gen) + " NN: " + str(i)
        print "Score: " + str(res[0]) + " Avg Food Dist: " + str(res[1])
        print "Fitness: " + str(fitnesses[i])
        if fitnesses[i] < bestFit:
            bestNN = population[i]
            bestFit = fitnesses[i]
            bestScore = res[0]
            bestGen = gen
            welp = 0
    
    if key == 27:
        break

    #go to next generation
    mostFit = getMostFit(fitnesses) 
    print "Gen: " + str(gen) + " Avg Fitness: " + str(mostFit[1])
    print "Creating next gen"
    
    fitInd = mostFit[0]
    mostFitNum = fitnesses[fitInd]
    mostFit[0] = population[fitInd]

    if welp == 20:
        mostFitNum = bestFit
        mostFit[0] = bestNN
    welp += 1    
    
    #swap the best nn to front
    fitnesses[fitInd] = fitnesses[0]
    fitnesses[0] = mostFitNum
    population[fitInd] = population[0]
    population[0] = mostFit[0]
    
    #make next generation
    for i in range(0, len(population)):
        #use average fitness to determine if should change
        if random.random() < (0.02+0.98*fitnesses[i]):
            #determine breeding or mutating
            if random.random() < 0.6:
                if random.random() < 0.8:
                    population[i] = breed(population[i], population[0], fitnesses[i])
                else:
                    ind = random.randint(0, len(population))
                    population[i] = breed(population[i], population[ind], max(fitnesses[i],fitnesses[ind]))
            elif random.random() < 0.3:
                population[i] = mutate(population[i], fitnesses[i], mostFit[1])
            else:
                population[i] = mutate(population[0], fitnesses[0], mostFit[1])

runGame(bestNN)
print "BestNN"
print bestNN.wi
print bestNN.wo
print bestScore
print "Best Gen: " + str(bestGen)