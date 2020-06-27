from environment import HaliteEnvironment
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from ddqn_agent import DDQNSolver

env = HaliteEnvironment(board_size = 5)

board = env.board

print(board.observation)
