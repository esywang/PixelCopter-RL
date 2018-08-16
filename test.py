from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import pygame
import numpy as np
import time

game = Pixelcopter(width=256, height=256)
p = PLE(game, fps=30, display_screen=False)

reward = 0.0
p.init()
game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)

while (True):
    p.reset_game()
    #time.sleep(1)
    while (not p.game_over()):
        action = np.random.randint(0, 2)
        actionList = p.getActionSet()
        reward = p.act(actionList[action])
        pygame.display.update()
