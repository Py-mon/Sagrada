from enum import Enum
from itertools import product
from math import factorial
from random import choices, randint, sample
from time import time
from typing import Literal, Optional, Self

import numpy as np
from scipy.special import softmax


def random(*shape):
    """Random numbers between -1 and +1"""
    return np.random.uniform(-1, 1, shape)


def inside_out(dct) -> dict:
    return {value: key for key, value in dct.items()}


Shade = Literal[1, 2, 3, 4, 5, 6]
Color = Literal[7, 8, 9, 10, 11]

ReprColor: dict[str, Color] = {"r": 7, "b": 8, "g": 9, "p": 10, "y": 11}
ColorRepr: dict[Color, str] = inside_out(ReprColor)

Null = Literal[0]

Restriction = Color | Shade | Null

Position = tuple[int, int]

Die = tuple[Color, Shade]


Tile = tuple[Color | Null, Shade | Null]  # list?  (prob bad to be list)


def restriction(tile: Tile):
    return tile[0] or tile[1]


def tile_die(tile: Tile) -> Optional[tuple[Color, Shade]]:
    if tile[0] != 0 and tile[1] != 0:
        return tile[0], tile[1]


def fulfills(die: Die, restriction: Restriction) -> bool:
    """Whether the dice can meet the requirements of the restriction."""
    return restriction == 0 or die[0] == restriction or die[1] == restriction


def can_fit(die: Die, tile: Tile) -> bool:
    """Whether the dice can meet the requirements of the tile (if any)."""
    return tile_die(tile) == 0 and fulfills(die, restriction(tile))


def make_grid(*tiles: Tile):
    grid = []
    for tile in tiles:
        grid += tile
    return grid


# TODO create chunk class


outer_tiles: list[Position] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (1, 0),
    (2, 0),
    (1, 4),
    (2, 4),
]


def adjacent(pos: Position):
    return [
        (pos[0] - 1, pos[1]),
        (pos[0] + 1, pos[1]),
        (pos[0], pos[1] - 1),
        (pos[0], pos[1] + 1),
    ]


def diagonal(pos: Position):
    return [
        (pos[0] - 1, pos[1] - 1),
        (pos[0] + 1, pos[1] + 1),
        (pos[0] + 1, pos[1] - 1),
        (pos[0] - 1, pos[1] + 1),
    ]


def get_legal_moves(grid, pool: list[Die]):
    # 3 for color, shade, pos, 6 shades, 5 colors, outer_tiles
    # legal = np.zeros(3 * 6 * 5 * len(outer_tiles))
    legal = np.zeros(13 * len(outer_tiles))  # ? how 13?
    grid = make_grid(*grid)
    for pos in outer_tiles:  # store outer_pos tiles
        tile = index_grid(grid, pos)
        illegals = set()
        for adjacent in tile_adjacent(pos):
            try:
                adjacent = index_grid(grid, adjacent)
            except IndexError:
                continue
            if restriction(adjacent) != 0:
                illegals.add(restriction(adjacent))

        for i, (die) in enumerate(pool):
            if fulfills(die, restriction(tile)) and all(
                not fulfills(die, illegal) for illegal in illegals
            ):
                legal[i * 3 : i * 3 + 3] = (*die, pos)
    return legal


def get_legal_positions(grid, pool: list[Die]):
    # 3 for color, shade, pos, 6 shades, 5 colors, outer_tiles
    # legal = np.zeros(3 * 6 * 5 * len(outer_tiles))
    legal = np.zeros(13 * len(outer_tiles))  # ? how 13?
    for y, x in outer_tiles:  # store outer_pos tiles
        tile = grid[y][x]
        illegals = set()
        for adjacent in adjacent((y, x)):
            try:
                adjacent = index_grid(grid, adjacent)
            except IndexError:
                continue
            if restriction(adjacent) != 0:
                illegals.add(restriction(adjacent))

        for i, (die) in enumerate(pool):
            if fulfills(die, restriction(tile)) and all(
                not fulfills(die, illegal) for illegal in illegals
            ):
                legal[i * 3 : i * 3 + 3] = (*die, pos)
    return legal


def make_grid_repr(tiles: list[list[str]]):
    grid = []
    for row in tiles:
        new_row = []
        for tile in row:
            if tile == " ":
                new_row.append((0, 0))
            elif ReprColor.get(tile):
                new_row.append((ReprColor.get(tile), 0))
            else:
                new_row.append((0, int(tile)))
        grid.append(new_row)
    return grid


grid = make_grid_repr(
    [
        ["g", " ", " ", " ", " "],
        [" ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " "],
    ]
)
print(grid)
print(get_legal_moves(grid, [(7, 2)]))


class GridNetwork:
    def __init__(self, height, width, input_size):
        self.height = height
        self.width = width
        self.input_size = input_size

        self.randomize_weights_biases()

    def randomize_weights_biases(self):
        self.biases = []
        # # Color
        # self.biases.append(random(self.height, self.width, 5))

        # Value
        self.biases.append(random(self.height, self.width))

        self.weights = []
        self.weights.append(random(self.input_size, self.height, self.width, 2))

    def flood_dice(self, legal_moves, grid):
        pass

# A
# legal_moves + grid = legal_move

# B
# Dice 1 -> 
# Grid
# o-x-o
# o/o\o
# o o o
# Filled Tiles get added to adjacent dice
# Dice 1 gets added to all legal positions
# highest firing position is winner
# highest firing dice with highest position is the networks choice

# all weights connected just not all used every time


# color = [0,0,0,0,1], [0,0,0,1,0] etc
