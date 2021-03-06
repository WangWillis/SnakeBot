# Snake code by sanchitgangwar
# Genetic Learning Alg code by Willis Wang
# SNAKES GAME
# Use ARROW KEYS to play, SPACE BAR for pausing/resuming and Esc Key for exiting

import copy
import tensorflow as tf
import numpy as np
import curses

import logging

from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint

np.set_printoptions(threshold=np.nan)

logging.basicConfig(filename="SnakeRLLog.log", level=logging.DEBUG)

screenX = 20
screenY = 60

KEYS = [KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN]
eps = 0.1

convF1Size = 20
convF2Size = 60
inConnW1Size = int((screenX/4)*(screenY/4)*convF2Size)
connW1Size = 5
connW2Size = 20
numOutputs = 4
def policy_network():
    with tf.variable_scope("policy"):
        gridIn = tf.placeholder("float32", [None, screenX, screenY, 1]) # used to hold grid input
        action = tf.placeholder("float32", [None, 4])
        advantages = tf.placeholder("float32", [None, 1])

        # set up conv nn
        convF1 = tf.get_variable("convF1", [5, 5, 1, convF1Size])
        convF2 = tf.get_variable("convF2", [5, 5, convF1Size, convF2Size])
        connW1 = tf.get_variable("connW1", [inConnW1Size, connW1Size])
        connW2 = tf.get_variable("connW2", [connW1Size, connW2Size])
        outLayer = tf.get_variable("outLayer", [connW2Size, numOutputs])

        #rnn definitions
        # rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=inConnW1Size,state_is_tuple=True)
        # state_in = rnn_cell.zero_state(inConnW1Size, tf.float32)
        # rnn, rnn_state = tf.nn.dynamic_rnn(inputs=convL2, cell=rnn_cell, dtype=tf.float32, initial_state=state_in)

        # model nn conv layer
        convL1 = tf.nn.conv2d(gridIn, convF1, strides=[1, 1, 1, 1], padding="SAME")
        convL1 = tf.nn.max_pool(convL1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        convL2 = tf.nn.conv2d(convL1, convF2, strides=[1, 1, 1, 1], padding="SAME")
        convL2 = tf.nn.max_pool(convL2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        #fully connected layer
        connIn = tf.reshape(convL2, [-1, inConnW1Size])
        connL1 = tf.matmul(connIn, connW1)
        connL1 = tf.nn.relu(connL1)
        connL2 = tf.matmul(connL1, connW2)
        connL2 = tf.nn.relu(connL2)
        preOut = tf.matmul(connL2, outLayer)
        poli_out = tf.nn.softmax(preOut)

        # advantage_ten = advantages-tf.reduce_mean(advantages, axis=0, keep_dims=True)
        # advantage_ten = advantages-tf.reduce_mean(advantages, axis=0, keep_dims=True)
        advantage_ten = advantages*action

        updateOut = preOut + advantage_ten

        cost = tf.square(updateOut-preOut)
        cost = tf.reduce_mean(cost)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

        return poli_out, gridIn, action, advantages, optimizer, updateOut, cost, advantage_ten

inL1Size = 5
hL1Size = 20
hL2Size = 10
outSize = 1
def value_network():
    with tf.variable_scope("value"):
        state = tf.placeholder("float32", [None, screenX*screenY])

        newvals = tf.placeholder("float",[None, 1])

        w1 = tf.get_variable("w1", [screenX*screenY, inL1Size])
        b1 = tf.get_variable("b1", [inL1Size])
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        w2 = tf.get_variable("w2", [inL1Size, hL1Size])
        b2 = tf.get_variable("b2", [hL1Size])
        w3 = tf.get_variable("w3", [hL1Size, hL2Size])
        b3 = tf.get_variable("b3", [hL2Size])

        calculated = tf.nn.relu(tf.matmul(h1,w2) + b2)
        calculated = tf.nn.relu(tf.matmul(calculated, w3) + b3)
        outW = tf.get_variable("outW", [hL2Size, outSize])
        outB = tf.get_variable("outB", [outSize])
        val_out = tf.matmul(calculated, outW) + outB

        diffs = newvals-val_out
        loss = tf.square(diffs)
        loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss


slow = False
noEps = False
key = 0
# nextKey = 0
doneKey = 0
foodScore = 100000
badScore = 0
#
# class SnakeGame:
#     def __init__():
#
#
#     def resetAndStart():
#         curses.initscr()
#         curses.noecho()
#         curses.curs_set(0)
#         self.win = curses.newwin(screenX, screenY, 0, 0)
#         self.win.keypad(1)
#         self.win.border(0)
#         self.win.nodelay(1)
#         doneKey = 0
#         key = KEY_RIGHT
#         score = 0
#         timeAlive = 0
#         timeLastScore = 0
#
#         #store the states and return for the learning
#         states = []
#         actions = []
#         transitions = []
#
#         results = []
#
#         grid = np.zeros((screenX, screenY))
#         grid.fill(badScore)
#         grid[1:-1,1:-1] = 0
#         snake = [[4,10], [4,9], [4,8]]                                     # Initial snake co-ordinates
#         grid[4][10] = badScore
#         grid[4][9] = badScore
#         grid[4][8] = badScore
#         food = []
#
#         while food == []:
#             foodX = randint(1, screenX-2)
#             foodY = randint(1, screenY-2)
#             if(grid[foodX][foodY] != 0):
#                 continue
#             food = [foodX, foodY]
#             grid[foodX][foodY] = foodScore
#
#         win.addch(food[0], food[1], '*')
#
#     def oneStep(move):


def runGame(poliOut, gridInput):
    global slow
    global noEps
    global key
    global doneKey
    global foodScore
    # global nextKey
    curses.initscr()
    curses.noecho()
    curses.curs_set(0)
    win = curses.newwin(screenX, screenY, 0, 0)
    win.keypad(1)
    win.border(0)
    win.nodelay(1)
    if(slow):
        win.timeout(1)
    doneKey = 0
    key = KEY_RIGHT                                                # Initializing values

    score = 0
    timeAlive = 0
    timeLastScore = 0

    #store the states and return for the learning
    states = []
    actions = []
    transitions = []

    results = []

    grid = np.zeros((screenX, screenY))
    # grid.fill(-50)
    # grid[1:-1,1:-1] = 0
    snakeX = randint(5, screenX-5)
    snakeY = randint(5, screenY-5)
    snake = [[snakeX,snakeY], [snakeX, snakeY-1], [snakeX,snakeY-2]]                                     # Initial snake co-ordinates
    grid[snakeX][snakeY] = -100
    grid[snakeX][snakeY-1] = -50
    grid[snakeX][snakeY-2] = -50

    food = []

    while food == []:
        foodX = randint(1, screenX-2)
        foodY = randint(1, screenY-2)
        if(grid[foodX][foodY] != 0):
            continue
        food = [foodX, foodY]
        grid[foodX][foodY] = 100

    win.addch(food[0], food[1], '*')

    while True:
        timeAlive += 0.01                                            # While Esc key is not pressed
        win.border(0)
        win.addstr(0, 2, 'Score: ' + str(score) + ' ')                # Printing 'Score' and

        # controls stuff
        action = [0, 0, 0, 0] #one hot vector representing which key pressed
        key = KEYS[3]
        if(not noEps and np.random.uniform(0,1) < eps):
            nextKey = randint(0, 3)
            action[nextKey] = 1
            key = KEYS[nextKey]
        else:
            gridIn = np.reshape(grid, [-1, screenX, screenY, 1])
            qValArr = sess.run(poliOut, feed_dict={gridInput: gridIn})[0]
            # logging.debug(qValArr)
            currMin = 0
            randNum = np.random.uniform(0,1)
            logging.debug(qValArr)
            for i in range(len(qValArr)):
                if(randNum >= currMin and randNum < qValArr[i]+currMin):
                    key = KEYS[i]
                    action[i] = 1
                    break
                currMin += qValArr[i]

        doneKey = win.getch()

        if(doneKey == ord("s")):
            slow = not slow
        if(doneKey == ord("n")):
            noEps = not noEps


        transition = [grid, action, 0]

        snake.insert(0, [snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1), snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1)])

        if snake[0] == food:                                            # When snake eats the food
            food = []
            score += 1
            transition[2] += foodScore
            timeLastScore = timeAlive

            while food == []:
                foodX = randint(1, screenX-2)
                foodY = randint(1, screenY-2)
                if(grid[foodX][foodY] != 0):
                    continue
                food = [foodX, foodY]
                grid[foodX][foodY] = 100

            win.addch(food[0], food[1], '*')
        else:
            last = snake.pop()                                          # [1] If it does not eat the food, length decreases
            win.addch(last[0], last[1], ' ')
            grid[last[0]][last[1]] = 0
        win.addch(snake[0][0], snake[0][1], '#')
        grid[snake[0][0]][snake[0][1]] = -100
        grid[snake[1][0]][snake[1][1]] = -50

        states.append(transition[0])
        actions.append(transition[1])
        transitions.append(transition)

        if(snake[0][0] <= 0 or snake[0][0] >= screenX-1 or snake[0][1] <= 0 or snake[0][1] >= screenY-1 or snake[0] in snake[1:]):
            transition[2] = 0
            break

        if (timeAlive-timeLastScore) > 5:
            break

        if(doneKey == 27):
            break

        transition[2] += 1
        key = win.getch()
    curses.endwin()

    return transitions, states, actions, score

policy_prob, grid_in, act, advantage, opt, updateOut, cost, advantage_ten = policy_network()
val_value, val_grid, val_correct, val_opt, val_loss = value_network()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

gameNum = 0
exploreTime = 100000
minExplore = 0.1
maxScore = 0
while doneKey != 27:
    transitions, states, actions, score = runGame(policy_prob, grid_in)
    advantages = []
    update_vals = []
    i = 0
    for transition in transitions:
        grid, action, reward = transition
        grid = np.reshape(grid, [-1, screenX*screenY])
        # calculate discounted monte-carlo return
        futReward = 0
        nextTranstions = len(transitions)-i
        gam = 1
        for j in range(nextTranstions):
            futReward += transitions[i+j][2] * gam
            gam = gam * 0.97

        val = sess.run(val_value, feed_dict={val_grid: grid})[0][0]

        # advantage: how much better was this action than normal
        advantages.append(futReward - val)

        # update the value function towards new return
        update_vals.append(futReward)
        i += 1
    # update value function
    states_reshape = np.reshape(states, [-1, screenX*screenY])
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    _, loss = sess.run([val_opt, val_loss], feed_dict={val_grid: states_reshape, val_correct: update_vals_vector})
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    states = np.reshape(states_reshape, [len(states), screenX, screenY, 1])
    advantages_vector = np.reshape(advantages, [len(advantages), 1])
    actions = np.reshape(actions, [len(actions), 4])
    probs = sess.run(policy_prob, feed_dict={grid_in:states})
    _, costVal = sess.run([opt, cost], feed_dict={grid_in: states, advantage: advantages_vector, act: actions})

    infoStr = "Game: " + str(gameNum) + " Score: " + str(score) + " MaxScore: " + str(maxScore) + " Val Cost: " + str(loss) + " Poly Cost: " + str(costVal) + " Eps: " + str(eps) + " No Eps: " + str(noEps)
    logging.info(infoStr)

    if(score > maxScore):
        maxScore = score

    # if(gameNum < exploreTime):
    #     eps = 1-((1-minExplore)/exploreTime)*gameNum
    gameNum += 1
sess.close()
