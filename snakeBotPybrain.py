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
        self.output = output
    
        #initialize the arrays for the layers
        self.il = [1.0] * self.inp
        self.hl = [1.0] * self.hidden
        self.ol = [1.0] * self.output

        #randomize weights
        self.wi = np.random.randn(self.inp, self.hidden)
        self.wo = np.random.randn(self.hidden, self.output)

    def think(self, inp):
        if len(inp) != self.inp-1:
            return ValueError("BAD INPUT")
    
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

        #propigate activations
        for j in range(self.output):
            tot = 0.0 
            #loop through hidden layer to change values
            for i in range(self.hidden):
                tot += self.hl[i]*self.wo[i][j]
            #set hidden layer values
            self.ol[j] = sigmoid(tot)

        return self.ol

screenX = 20
screenY = 60
inputSize = 2
numHiddenLayers = 500
numOutputs = 5
numChrome = 8
gen = 0
key = 0
KEYS = [KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN, KEY_RIGHT]

fitnesses = [0] * numChrome
population = [0] * numChrome

bestNN = 0
bestScore = -1

def createScreen(snake, food):
    screen = [0]*inputSize

    for i in range(0, len(snake)):
        screen[snake[i][0]*screenY+snake[i][1]] = 5*(snake[i][0]+snake[i][1])
    screen[snake[0][0]*screenY+snake[0][1]] = 20*(snake[0][0]+snake[0][1])
    screen[food[0]*screenY+food[1]] = 50*(snake[0][0]+snake[0][1])
 
    
    return screen

def getBiggestIndex(arr):
    index = 0
    for i in range(1, len(arr)):
        if arr[i] > arr[index]:
            index = i
    return index

def runGame(nn):
    curses.initscr()
    curses.noecho()
    curses.curs_set(0)
    win = curses.newwin(screenX, screenY, 0, 0)
    win.keypad(1)
    win.border(0)
    win.nodelay(1)
    global key
    key = KEY_RIGHT                                                    # Initializing values
    score = 0
    timeAlive = 0
    timeLastScore = 0 
    avgDist = 0
    results = []

    snake = [[4,10], [4,9], [4,8]]                                     # Initial snake co-ordinates
    food = []                                                     # First food co-ordinates

    while food == []:
        food = [randint(1, 18), randint(1, 58)]                 # Calculating next food's coordinates
        if food in snake: food = []
    win.addch(food[0], food[1], '*')

    while key != 27:                                                   # While Esc key is not pressed
        win.border(0)
        win.addstr(0, 2, 'Score: ' + str(score) + ' ')                # Printing 'Score' and
        win.addstr(0, 15, 'Gen: ' + str(gen) + ' ')
        win.addstr(0, 30, 'Sec: ' + str(timeAlive) + ' ')
        win.addstr(0, 27, ' SNAKE ')                                   # 'SNAKE' strings
        win.timeout(5)
        timeAlive += 0.01
        xDis = snake[0][0]-food[0]
        yDis = snake[0][1]-food[1]
        avgDist += math.sqrt(xDis*xDis+yDis*yDis)
        prevKey = key
        KEYS[4] = prevKey
        event = KEYS[getBiggestIndex(nn.think([xDis, yDis]))]
        done = win.getch()
        key = done if done == 27 else event 

        if key == ord(' '):                                            # If SPACE BAR is pressed, wait for another
            key = -1                                                   # one (Pause/Resume)
            while key != ord(' '):
                key = win.getch()
            key = prevKey
            continue

        if key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, 27]:     # If an invalid key is pressed
            key = prevKey
        
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
        win.addch(snake[0][0], snake[0][1], '#')
        if (timeAlive-timeLastScore) > 3:
            break
        
    results.append(score)
    results.append(timeAlive)
    results.append((timeAlive*100)/avgDist)
    curses.endwin()
    return results

def breed(nn1, nn2, fit1, fit2):
    random.seed()
    prob = random.random()
    if fit1 > 0 and fit2 > 0:
        prob = fit2/(fit1*2)
    nnNew = copy.deepcopy(nn1)

    size = nnNew.wi.shape 
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wi[i][j] = nn2.wi[i][j]
    
    size = nnNew.wo.shape 
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wo[i][j] = nn2.wo[i][j]
    return nnNew

def mutate(nn, fit):
    random.seed()
    nnNew = copy.deepcopy(nn)
    prob = 0.4

    size = nnNew.wi.shape
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wi[i][j] = random.uniform((-2)*nnNew.wi[i][j], nnNew.wi[i][j]*2)

    size = nnNew.wo.shape
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if random.random() < prob:
                nnNew.wo[i][j] = random.uniform((-2)*nnNew.wo[i][j], nnNew.wo[i][j]*2)

    return nnNew

def getMostFit(fitnesses):
    random.seed()
    mostFit = [0, 0, 0.0]
    for i in range(len(fitnesses)):
        mostFit[2] += fitnesses[i]
        if fitnesses[i] > fitnesses[mostFit[0]]:
            mostFit[1] = mostFit[0]
            mostFit[0] = i
        if fitnesses[i] == fitnesses[mostFit[0]] and random.random() < 0.5:
            mostFit[1] = mostFit[0]
            mostFit[0] = i

    mostFit[2] /= len(fitnesses)
    return mostFit

def fitness(res):
    return res[0]*res[0]+res[1]+2*res[2]

#init starting off neural networks
for i in range(len(population)):
    population[i] = NeuralNetwork(inputSize, numHiddenLayers, numOutputs)

random.seed()

#loop for 1000 generations
while key != 27:
    gen += 1
    #play game for whole population
    for i in range(len(population)):
        res = runGame(population[i])
        fitnesses[i] = fitness(res)

        if key == 27:
            break
       
        #print data
        print "Gen: " + str(gen) + " NN: " + str(i)
        print "Score: " + str(res[0]) + " Avg Food Dist: " + str(res[1])
        print "Fitness: " + str(fitnesses[i])

        if res[0] > bestScore:
            bestNN = population[i]
            bestScore = res[0]
    
    if key == 27:
        break

    #go to next generation
    mostFit = getMostFit(fitnesses) 
    print "Gen: " + str(gen) + " Avg Fitness: " + str(mostFit[2])
    print "Creating next gen"
    
    bestFit = fitnesses[mostFit[0]]
    secFit = fitnesses[mostFit[1]]

    mostFit[0] = population[mostFit[0]]
    mostFit[1] = population[mostFit[1]]
    population[0] = mostFit[0] 
    #make next generation
    for i in range(1, len(population)):
        #determine mutate or breed 
        if random.random() < 0.4:
            population[i] = breed(mostFit[0], mostFit[1], bestFit, secFit)
        else:
            #determine if mutate most fit or second most
            if random.random() < 0.6:
                population[i] = mutate(mostFit[0], bestFit)
            else:
                population[i] = mutate(mostFit[1], secFit)
runGame(bestNN)
print "BestNN"
print bestNN.wi
print bestNN.wo
print bestScore
