from queue import PriorityQueue
from typing import Dict, Tuple
from collections import defaultdict
from itertools import product
from scipy.stats import norm

import cv2
import numpy as np
from matplotlib import pyplot as plt
from environment import Environment
from utils import generate_maze


def a_star_search(occ_map: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Implements the A* search with heuristic function being distance from the goal position.
    :param occ_map: Occupancy map, 1 – field is occupied, 0 – is not occupied.
    :param start: Start position from which to perform search
    :param end: Goal position to which we want to find the shortest path
    :return: The dictionary containing at least the optimal path from start to end in the form:
        {start: intermediate, intermediate: ..., almost: goal}
    """
    """ TODO: your code goes here """
    def distance(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** (1/2)

    def get_occupancy(x, y):
        # We assume occ_map is indexed (x, y).
        if x < 0 or x >= occ_map.shape[0] or y < 0 or y >= occ_map.shape[1]:
            return 1
        else:
            return occ_map[x, y]

    def revert_came_from(came_from):
        result = dict()
        current = end
        while current != start:
            previous = came_from[current]
            result[previous] = current
            current = previous
        return result
    
    if get_occupancy(*start) == 1:
        return None

    open_set = PriorityQueue()
    
    came_from = dict()
    g_score = defaultdict(lambda: float('inf'))
    f_score = defaultdict(lambda: float('inf'))

    g_score[start] = 0
    f_score[start] = distance(start, end)
    open_set.put((f_score[start], start))

    while not open_set.empty():
        current = open_set.get()[1]
        if current == end:
            return revert_came_from(came_from)
        for neighbour in [(current[0] + i, current[1] + j) for i,j in [(0, 1), (0, -1), (1, 0), (-1, 0)]]:
            if get_occupancy(*neighbour) == 0:
                if g_score[current] + 1 < g_score[neighbour]:
                    came_from[neighbour] = current
                    g_score[neighbour] = g_score[current] + 1
                    f_score[neighbour] = g_score[current] + 1 + distance(neighbour, end)
                    if neighbour not in map(lambda x: x[1], open_set.queue):
                        open_set.put((f_score[neighbour], neighbour))
                    
    return None # No path from start to end.

def get_delta(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    result = tuple(np.array(b) - np.array(a))
    assert result in [(0, 1), (0, -1), (1, 0), (-1, 0)], f"Incorrect delta: {result}"
    return result


class LocalizationMap:
    def __init__(self, environment):
        """ TODO: your code goes here """
        self.env = environment
        self.maze_size = 11
        self.cell_size = 10
        resolution = self.maze_size * self.cell_size
        self.shape = (resolution, resolution)
        self.bel = 1 - self.env.gridmap.copy()
        self.bel /= self.bel.sum()

    def position_update_by_motion_model(self, delta: np.ndarray) -> None:
        """
        :param delta: Movement taken by agent in the previous turn.
        It should be one of [[0, 1], [0, -1], [1, 0], [-1, 0]]
        """
        """ TODO: your code goes here """
        new_bel = self.env.position_stochasticity * self.bel.copy()
        delta_x, delta_y = delta
        
        n = self.bel.shape[0]
        d = {0: slice(n), 1:slice(n-1), -1:slice(1,n)}
        y_slice1, y_slice2, x_slice1, x_slice2 = map(lambda x: d[x], [delta_y, -delta_y, delta_x, -delta_x])

        possible_part = self.bel[x_slice1, y_slice1]
        new_bel[x_slice2, y_slice2] += (1 - self.env.position_stochasticity) * possible_part
        
        # We take into consideration the fact that we cannot be inside the wall.
        new_bel[self.env.gridmap == 1] = 0 
        new_bel /= new_bel.sum()

        assert abs(np.sum(new_bel) - 1.) < 0.0001
        self.bel = new_bel


    def position_update_by_measurement_model(self, distances: np.ndarray) -> None:
        """
        Updates the probabilities of agent position using the lidar measurement information.
        :param distances: Noisy distances from current agent position to the nearest obstacle.
        """
        """ TODO: your code goes here """
        for x, y in product(range(self.shape[0]), range(self.shape[1])):
            if not self.env.gridmap[x, y]:
                _, true_distances = self.env.ideal_lidar(self.env.rowcol_to_xy((x, y)))
                self.bel[x, y] = np.prod(norm.pdf(distances / true_distances, loc=1, scale=self.env.lidar_stochasticity)) * self.bel[x, y]
        self.bel /= np.sum(self.bel)

    def position_update(self, distances: np.ndarray, delta: np.ndarray = None):
        self.position_update_by_motion_model(delta)
        self.position_update_by_measurement_model(distances)


class LocalizationAgent:
    def __init__(self, environment):
        """ TODO: your code goes here """
        self.map = LocalizationMap(environment)
        self.env = environment
        self.paths = None

        # Using position just to fix bug of spawning in occupied position,
        # as discussed on the forum.
        starting_position = self.env.xy_to_rowcol(self.env.position())
        if self.env.gridmap[starting_position[0], starting_position[1]] == 1:
            raise ValueError('Starting position is occupied!')

    def step(self) -> None:
        """
        Localization agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the agent position probabilities
            * choosing and executing the next agent action in the environment
        """
        """ TODO: your code goes here """
        _, distances = self.env.lidar()
        self.map.position_update_by_measurement_model(distances)
        
        best_guess = np.unravel_index(self.map.bel.argmax(), self.map.bel.shape)

        if not self.paths or best_guess not in self.paths:
            self.paths = a_star_search(self.env.gridmap, best_guess, self.env.xy_to_rowcol(self.env.goal_position))
        
        if self.paths is None or best_guess not in self.paths:
            # If agent get stuck it performs randomly.
            delta = [(0, 1), (0, -1), (1, 0), (-1, 0)][np.random.randint(4)]
        else:
            delta = get_delta(best_guess, self.paths[best_guess])
        
        self.env.step(delta)
        self.map.position_update_by_motion_model(delta)

    def visualize(self) -> np.ndarray:
        """
        :return: the matrix of probabilities of estimation of current agent position
        """
        """ TODO: your code goes here """
        return self.map.bel


if __name__ == "__main__":
    maze = generate_maze((11, 11))
    env = Environment(
        maze,
        lidar_angles=3,
        resolution=1/11/10,
        agent_init_pos=None,
        goal_position=(0.87, 0.87),
        position_stochasticity=0.5
    )
    agent = LocalizationAgent(env)

    while not env.success():
        agent.step()

        if env.total_steps % 10 == 0:
            plt.imshow(agent.visualize())
            plt.colorbar()
            plt.savefig('/tmp/map.png')
            plt.close(plt.gcf())

            cv2.imshow('map', cv2.imread('/tmp/map.png'))
            cv2.waitKey(1)

    print(f"Total steps taken: {env.total_steps}, total lidar readings: {env.total_lidar_readings}")
