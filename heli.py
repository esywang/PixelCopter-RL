import theano
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import pygame
import numpy as np
import time
from IPython.display import clear_output
import random
import matplotlib.pyplot as plt
import datetime

model = Sequential()
model.add(Dense(20, input_dim=5, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(2, activation="linear"))
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer="adam")

#model = load_model("weights/w_2018-08-14_22:58:01.640725")

game = Pixelcopter(width=256, height=256)
p = PLE(game, fps=30, display_screen=False)

reward = 0.0
p.init()
game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
#game.clock = pygame.time.Clock()
count = 0

epochs = 100000
rewards = np.zeros((1,epochs))[0]
gamma = 0.99 #since it may take several moves to goal, making gamma high
epsilon = 0.75
batchSize = 200
buffer = 400
replay = []
i = 0
h = 0

while (i < epochs):
    
    p.reset_game()
    
    while (not p.game_over()):
        state = game.getGameState()
        stateLst = np.array([[state[k] for k in state]])
        qval = model.predict(stateLst)
        if (random.random() < epsilon):
            action = np.random.randint(0,2)
            print("here", action)
        else:

        	
            action = np.argmax(qval)
            
        actionList = p.getActionSet()
        reward = p.act(actionList[action])

        newState = game.getGameState()
        newStateLst = np.array([[newState[k] for k in state]])

        if (len(replay) < buffer):
            replay.append((stateLst, action, reward, newStateLst))
        else:
            if h < buffer-1:
                h += 1
            else:
                h = 0
            replay[h] = (stateLst, action, reward, newStateLst)
            minibatch = random.sample(replay, batchSize)
            X_train = np.empty((0,5))
            y_train = np.empty((0,2))

            for memory in minibatch:
                old_state, action, reward, new_state = memory
                oldQ = model.predict(old_state)
                newQ = model.predict(newStateLst)
                maxQ = np.max(newQ)

                if p.game_over():
                    update = reward
                else:
                    update = reward + (gamma * maxQ)
                y = np.copy(oldQ)
                y[0][action] = update
                X_train = np.append(X_train, old_state, axis=0)
                y_train = np.append(y_train, y, axis=0)

            model.fit(X_train, y_train, batch_size=batchSize)

        pygame.display.update()
        print(i)
        clear_output(wait=True)
        
    rewards[i] = p.score()
    i += 1
    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1/epochs)
    
    if i % 10000 == 0:    
        f = "weights/w_" + str(datetime.datetime.now()).replace(" ", "_")
        model.save(f)
