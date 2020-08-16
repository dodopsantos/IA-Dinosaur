#!/usr/bin/python

import time

from PIL import ImageGrab
import pyautogui
import neat
import os
import pygame

X = 150
X2 = 450
Y1 = 390
Y2 = 470

# over = x(890, 290)

def capture_screen():
    screen = ImageGrab.grab()
    return screen

def detect_enemy(screen):
    aux_color = screen.getpixel((int(X), Y1))
    for x in range(int(X), int(X2)):
        for y in range(Y1, Y2):
            color = screen.getpixel((x, y))
            if color != aux_color:
                return (x - 150, 1)
            else:
                aux_color = color
                return (999, 0)


def morte(screen):
    pyautogui.press("space")
    return screen.getpixel((890, 290)) == (83,83,83)

def jump():
    pyautogui.press("up")

def down():
    pyautogui.press("down")

print("Start in 3 seconds...")
time.sleep(3)


def evalfitness(genomes, config):

    nets = []
    ge = []

    for g_id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        ge.append(g)
        # ge.pop(1)
    ge.pop(1)

    while True:
       time.sleep(0.2)
       ge[0].fitness += 0.1
       screen = capture_screen()
       output = nets[0].activate(detect_enemy(screen))

       if (output[0] > 0.5):
           jump()
       if (output[1] > 0.5):
           down()

       if morte(screen) :
           ge[0].fitness -= 5
           ge.pop(0)
           nets.pop(0)
           break


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(evalfitness, 500)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
