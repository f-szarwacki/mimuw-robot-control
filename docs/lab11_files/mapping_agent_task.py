from itertools import product
from typing import Tuple, Optional

import cv2

import numpy as np
from matplotlib import pyplot as plt

from environment import Environment
from localization_agent_task import a_star_search, get_delta
from utils import generate_maze, bresenham


class OccupancyMap:
    def __init__(self, environment):
        """ TODO: your code goes here """
        self.env = environment
        self.log_odds = np.zeros_like(self.env.gridmap)
        self.l_occ = 1
        self.l_free = -0.5
        self.eps = 0.002

    def point_update(self, pos: Tuple[int, int], distance: Optional[float], total_distance: Optional[float], occupied: bool) -> None:
        """
        Update regarding noisy occupancy information inferred from lidar measurement.
        :param pos: rowcol grid coordinates of position being updated
        :param distance: optional distance from current agent position to the :param pos: (your solution don't have to use it)
        :param total_distance: optional distance from current agent position to the final cell from current laser beam (your solution don't have to use it)
        :param occupied: whether our lidar reading tell us that a cell on :param pos: is occupied or not
        """
        """ TODO: your code goes here """
        x, y = pos

        if occupied:
            self.log_odds[x, y] = min(self.log_odds[x, y] + self.l_occ, 10)
        else:
            self.log_odds[x, y] = max(self.log_odds[x, y] + self.l_free, -5)

    def map_update(self, pos: Tuple[float, float], angles: np.ndarray, distances: np.ndarray) -> None:
        """
        :param pos: current agent position in xy in [0; 1] x [0; 1]
        :param angles: angles of the beams that lidar has returned
        :param distances: distances from current agent position to the nearest obstacle in directions :param angles:
        """
        """ TODO: your code goes here """
        pos = np.array(pos)
        for angle, distance in zip(angles, distances):
            end = pos + np.array([np.cos(angle), np.sin(angle)]) * distance # TODO check angles
            points_to_check = bresenham(self.env.xy_to_rowcol(pos), self.env.xy_to_rowcol(end))
            last_idx = len(points_to_check) - 1
            for i, (x, y) in enumerate(points_to_check):
                if x < self.log_odds.shape[0] and y < self.log_odds.shape[1]:
                    v = np.array(self.env.rowcol_to_xy((x, y)))
                    r = np.linalg.norm(v - pos)
                    self.point_update((x, y), r, distance, i == last_idx)


class MappingAgent:
    def __init__(self, environment):
        """ TODO: your code goes here """
        self.env = environment
        self.map = OccupancyMap(environment)
        self.steps_to_be_made = 0
        self.delta_to_be_used = None

        # Using gridmap just to fix bug of spawning in occupied position,
        # as discussed on the forum.
        starting_position = self.env.xy_to_rowcol(self.env.position())
        if self.env.gridmap[starting_position[0], starting_position[1]] == 1:
            raise ValueError('Starting position is occupied!')

    def step(self) -> None:
        """
        Mapping agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the occupancy map beliefs/probabilities about their state
            * choosing and executing the next agent action in the environment
        """
        """ TODO: your code goes here """
        if self.steps_to_be_made > 0 and self.delta_to_be_used is not None:
            self.steps_to_be_made -= 1
            self.env.step(self.delta_to_be_used)
            return
        
        angles, distances = self.env.lidar()
        self.map.map_update(self.env.position(), angles, distances)
        
        current_pos = self.env.xy_to_rowcol(self.env.position())
        
        maze_width = int(1. / self.env.cell_length)
        cell_width = self.map.log_odds.shape[0] // maze_width

        # At the beginning we work with whole maze cells to make it less random.
        current_pos_scaled = tuple(map(lambda x: x // cell_width, current_pos))
        goal_pos_scaled = tuple(map(lambda x: x // cell_width, self.env.xy_to_rowcol(self.env.goal_position)))

        if current_pos_scaled != goal_pos_scaled:
            # We assume that a maze cell is occupied if at least 10% of its points have
            # > 50% probability of being occupied.
            current_knowledge_map = (np.mean(self.map.log_odds.reshape(
                (maze_width, cell_width, maze_width, cell_width)), 
                axis=(1,3)) > 0.05
            ).astype(int)
            
            paths = a_star_search(
                current_knowledge_map, 
                current_pos_scaled, 
                goal_pos_scaled
            )
            
            if paths is None or np.random.uniform() < 0.1:
                # If agent get stuck it performs randomly.
                delta = [(0, 1), (0, -1), (1, 0), (-1, 0)][np.random.randint(4)]
            else:
                delta = get_delta(current_pos_scaled, paths[current_pos_scaled])
            
            # We use a bigger move to get between maze cells.
            self.steps_to_be_made = 4
            self.delta_to_be_used = delta
            self.env.step(delta)
        else:
            # When in correct maze cell we become more precise to get to specific point.
            paths = a_star_search(
                (self.visualize() > 0.5).astype(int), 
                current_pos, 
                self.env.xy_to_rowcol(self.env.goal_position)
            )

            if paths is None or np.random.uniform() < 0.1:
                # If agent get stuck it performs randomly.
                delta = [(0, 1), (0, -1), (1, 0), (-1, 0)][np.random.randint(4)]
            else:
                delta = get_delta(current_pos, paths[current_pos])
            self.env.step(delta)

    def visualize(self) -> np.ndarray:
        """
        :return: the matrix of probabilities of estimation of given cell occupancy
        """
        """ TODO: your code goes here """
        return 1. - 1. / (1. + np.exp(self.map.log_odds))


if __name__ == "__main__":
    maze = generate_maze((11, 11))

    env = Environment(
        maze,
        resolution=1/11/10,
        agent_init_pos=(0.136, 0.136),
        goal_position=(0.87, 0.87),
        lidar_angles=256
    )
    agent = MappingAgent(env)

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
