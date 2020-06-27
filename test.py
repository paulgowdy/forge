import numpy as np
from ddqn_agent import DDQNSolver
from environment import HaliteEnvironment
import time
from environment import HaliteEnvironment
import random
import math
import numpy as np
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1,2,3,4,4,2,0])
plt.show()

game_name = "Halite"
BOARD_SIZE = 5
INPUT_SHAPE = (BOARD_SIZE,BOARD_SIZE,3)
ACTION_SPACE_SIZE = 5

test_eps = 10
max_ep_steps = 100

run = 0

game_model = DDQNSolver(game_name, INPUT_SHAPE, ACTION_SPACE_SIZE)
env = HaliteEnvironment(BOARD_SIZE)


class BoardImageRepresentation:

    def __init__(self):

        self.__cmap = {'empty': np.array([0,0,0]), # empty cells

                       'hlt_75_100': np.array([0,0,255]), # cells with halite
                       'hlt_50_75': np.array([0,0,212]),
                       'hlt_25_50': np.array([0,0,170]),
                       'hlt_0_25': np.array([0,0,128]),

                       'main_plr_ship': np.array([255,0,0]), # cells with player units
                       'main_plr_current_ship': np.array([255,0,128]),
                       'main_plr_yard': np.array([128,0,0]),
                       'main_plr_current_yard': np.array([128,0,128]),

                       'other_plr_ship': np.array([0,255,0]), # cells with enemies
                       'other_plr_yard': np.array([0,128,0])}


    def represent(self, board):
        gen_view = self.__get_general_view(board)
        highlighted_ships = self.__get_highlighted_ships(board, gen_view)
        highlighted_shipyards = self.__get_highlighted_shipyards(board, gen_view)

        board_img = {'general_view': gen_view,
                     'highlighted_ships': highlighted_ships,
                     'highlighted_shipyards': highlighted_shipyards}

        board_img = self.__rotate_board_img(board_img)
        board_img = self.__normalize_board_img(board_img)

        return board_img


    def render(self, board_img):
        plt.clf()
        plt.subplot(1,1,1)
        plt.imshow(board_img['general_view'])
        plt.axis('off')
        plt.title(f"General view", fontsize=20)
        #plt.show()
        plt.pause(0.1)
        '''
        ships_count = len(board_img['highlighted_ships'])
        if ships_count > 0:
            row_count = math.ceil(ships_count / 3)
            plt.figure(figsize=(6*3,5*row_count))
            for i, (ship_id, mtx) in enumerate(board_img['highlighted_ships'].items()):
                ax = plt.subplot(row_count,3,i+1)
                ax.imshow(mtx)
                plt.axis('off')
                plt.title(f"Ship ID: {ship_id}", fontsize=20)
            plt.show()

        shipyards_count = len(board_img['highlighted_shipyards'])
        if shipyards_count > 0:
            row_count = math.ceil(shipyards_count / 3)
            plt.figure(figsize=(6*3,5*row_count))
            for i, (shipyard_id, mtx) in enumerate(board_img['highlighted_shipyards'].items()):
                ax = plt.subplot(row_count,3,i+1)
                ax.imshow(mtx)
                plt.axis('off')
                plt.title(f"Shipyard ID: {shipyard_id}", fontsize=20)
            plt.show()
        '''

    def __get_general_view(self, board):
        board_size = board.configuration.size
        max_cell_halite = board.configuration.max_cell_halite
        main_plr_id = board.current_player_id

        gen_view = np.zeros((board_size,board_size,3))

        for coords, cell in board.cells.items():
            if cell.ship is not None:
                plr_role = 'main' if cell.ship.player_id == main_plr_id else 'other'
                gen_view[coords] = self.__cmap[f'{plr_role}_plr_ship']

            elif cell.shipyard is not None:
                plr_role = 'main' if cell.shipyard.player_id == main_plr_id else 'other'
                gen_view[coords] = self.__cmap[f'{plr_role}_plr_yard']

            elif cell.halite > 0:
                hlt_percent = cell.halite / max_cell_halite * 100
                hlt_interval = self.__get_hlt_percent_interval(hlt_percent)
                gen_view[coords] = self.__cmap[f'hlt_{hlt_interval}']

        return gen_view


    def __get_highlighted_ships(self, board, general_view):
        highlighted_ships = dict()

        for ship in board.current_player.ships:
            gen_view_cp = general_view.copy()
            gen_view_cp[ship.position] = self.__cmap['main_plr_current_ship']
            highlighted_ships[ship.id] = gen_view_cp

        return highlighted_ships


    def __get_highlighted_shipyards(self, board, general_view):
        highlighted_shipyards = dict()

        for shipyard in board.current_player.shipyards:
            gen_view_cp = general_view.copy()
            gen_view_cp[shipyard.position] = self.__cmap['main_plr_current_yard']
            highlighted_shipyards[shipyard.id] = gen_view_cp

        return highlighted_shipyards


    def __get_hlt_percent_interval(self, hlt_percent):
        interval_dict = {(0,25):'0_25', (25,50):'25_50', (50,75):'50_75', (75,np.inf):'75_100'}
        for interval in interval_dict.keys():
            if interval[0] < hlt_percent <= interval[1]:
                return interval_dict[interval]


    def __apply_func_to_board_img(self, board_img, func):
        board_img['general_view'] = func(board_img['general_view'])

        for ship_id, mtx in board_img['highlighted_ships'].items():
            board_img['highlighted_ships'][ship_id] = func(mtx)

        for shipyard_id, mtx in board_img['highlighted_shipyards'].items():
            board_img['highlighted_shipyards'][shipyard_id] = func(mtx)

        return board_img


    def __normalize_board_img(self, board_img):
        func = lambda x: np.round(x / 255.0, 3)
        return self.__apply_func_to_board_img(board_img, func)


    def __rotate_board_img(self, board_img):
        func = lambda x: np.rot90(x)
        return self.__apply_func_to_board_img(board_img, func)


board = env.board

board_img_repr = BoardImageRepresentation()
board_img = board_img_repr.represent(board)


for test_ep in range(test_eps):
    print("Test Ep:", test_ep)

    current_state = env.reset()

    plt.figure(figsize=(5,5))

    for step in range(max_ep_steps):
        '''
        action = game_model.move(current_state)
        next_state, reward, terminal, info = env.step(action)

        current_state = next_state

        board_img = board_img_repr.represent(env.board)
        board_img_repr.render(board_img)
        '''
        action = random.randint(0,4)
        print(action)
        env.step(action)
        board_img = board_img_repr.represent(env.board)
        board_img_repr.render(board_img)

    score = env.board.observation['players'][0][0]
    print(score)
    print("")
