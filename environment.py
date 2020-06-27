import numpy as np

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

class HaliteEnvironment:

    def __init__(self, board_size=5, startingHalite=1000):

        self.agent_count = 1
        self.board_size = board_size
        self.max_nb_ships = 1

        self.environment = make("halite", configuration={"size": board_size, "startingHalite": startingHalite})
        self.environment.reset(self.agent_count)

        state = self.environment.state[0]
        self.board = Board(state.observation, self.environment.configuration)

        #self.max_steps = 400
        #self.current_step = 0

        # Ship actions in order:
        # [Hold, North, East, South, West, Convert]
        self.ship_action_conversion_dict = {
        0:None,
        1:ShipAction.NORTH,
        2:ShipAction.EAST,
        3:ShipAction.SOUTH,
        4:ShipAction.WEST,
        5:ShipAction.CONVERT
        }

    '''
    def step(self, ship_actions, yard_actions):

        ship_actions_converted = self.convert_ship_actions(ship_actions)
        # just leave yard_actions as binary, if 1, then make a ship
        # also, have precleaned these before passing here
        # match current number of ships and yards

        for a, ship_id in zip(ship_actions_converted, iter(self.board.observation['players'][0][2])):
            #print(ship_id)
            self.board.ships[ship_id].next_action = a

        for a, yard_id in zip(yard_actions, iter(self.board.observation['players'][0][1])):
            if a == 1:
                self.board.shipyards[yard_id].next_action = ShipyardAction.SPAWN

        self.board = self.board.next()
    '''

    def step(self, ship_action, max_ep_steps):

        initial_halite = self.board.observation['players'][0][0]

        if ship_action == -1:
            for yard_id in iter(self.board.observation['players'][0][1]):
                self.board.shipyards[yard_id].next_action = ShipyardAction.SPAWN

        else:
            ship_action_converted = self.convert_ship_actions(ship_action)
            for a, ship_id in zip(ship_action_converted, iter(self.board.observation['players'][0][2])):
                self.board.ships[ship_id].next_action = a

        self.board = self.board.next()

        post_move_halite = self.board.observation['players'][0][0]

        next_state = self.board_to_obs(self.board)
        reward = (post_move_halite - initial_halite)/100.
        terminal = False
        if self.board.observation['step'] == max_ep_steps:
            terminal = True
        info = False

        return next_state, reward, terminal, info

    def convert_ship_actions(self, single_ship_action):
        # Ship actions in order:
        # [Hold, North, East, South, West, Convert]
        return [self.ship_action_conversion_dict[single_ship_action]]

    '''
    def count_elements(self):
        nb_ships = len(self.board.current_player.ships)
        nb_yards = len(self.board.current_player.shipyards)

        return nb_ships, nb_yards
    '''
    def board_to_obs(self, board):
        # normalize inputs here, well theyre not floats exactly

        raw_obs = board.observation

        halite_layer = raw_obs['halite']
        #yard_layer = [0] * len(halite_layer)
        ship_layer = [0] * len(halite_layer)

        halite_layer = np.array(halite_layer)
        halite_layer = halite_layer.reshape((self.board_size, self.board_size))
        halite_layer = halite_layer/200.

        #player_yards = raw_obs['players'][0][1]
        player_ships = raw_obs['players'][0][2]

        '''
        for yard_loc in list(player_yards.values()):
            yard_layer[yard_loc] = 1
        '''

        for ship in list(player_ships.values()):
            ship_layer[ship[0]] = (ship[1] + 1)/200.

        # Will need to eventuall include other players...
        # probably seperate layers for each player
        #yard_layer = np.array(yard_layer)
        #yard_layer = yard_layer.reshape((self.board_size, self.board_size))

        ship_layer = np.array(ship_layer)
        ship_layer = ship_layer.reshape((self.board_size, self.board_size))

        ship_select_layers = np.zeros((self.max_nb_ships, self.board_size, self.board_size))
        #yard_select_layers = np.zeros((self.max_nb_yards, self.board_size, self.board_size))

        self.nb_ships = len(list(player_ships.values()))
        if self.nb_ships > self.max_nb_ships:
            self.nb_ships = self.max_nb_ships
        #self.nb_yards = len(list(player_yards.values()))
        #if self.nb_yards > self.max_nb_yards:
        #    self.nb_yards = self.max_nb_yards

        for i in range(self.nb_ships):
            #print(ship_select_layers.shape)
            ship_1d_coord = list(player_ships.values())[i][0]

            col = int(ship_1d_coord % self.board_size)
            row = int((ship_1d_coord - col) / self.board_size)

            ship_select_layers[i, row, col] = 1
        '''
        for i in range(self.nb_yards):
            #print(ship_select_layers.shape)
            #print(i, player_yards.values())
            yard_1d_coord = list(player_yards.values())[i]

            col = int(yard_1d_coord % self.board_size)
            row = int((yard_1d_coord - col) / self.board_size)

            yard_select_layers[i, row, col] = 1
        '''

        #print('halite_layer')
        #print(halite_layer)
        #print('ship_layer')
        #print(ship_layer)
        ##print('yard_layer')
        ##print(yard_layer)

        #print('ship_select_layers')
        #print(ship_select_layers[0,:,:])
        ##print(ship_select_layers[1,:,:])
        ##print(ship_select_layers[2,:,:])


        board_layers = np.array([halite_layer, ship_layer])
        # will need to add yard layer back in here eventually
        # also, how to distinguish between players...
        # if seperate 'board' layer for each players ships and yards
        # gonna get biiiiig

        obs = np.concatenate([board_layers, ship_select_layers], 0)
        board_obs = np.moveaxis(obs, 0, 2)
        #print(obs.shape)
        scalar_obs = [raw_obs['step']/100., (raw_obs['players'][0][0] - 4000.)/1000.]
        #print(scalar_obs)

        return [board_obs, scalar_obs]

    def reset(self):
        self.environment.reset(self.agent_count)
        state = self.environment.state[0]
        self.board = Board(state.observation, self.environment.configuration)
        return self.board_to_obs(self.board)
