import heapq
import time

import pygame
from typing import List
from utils import Colors
from enum import Enum, auto
from functools import partial
from itertools import permutations


class NodeType(Enum):
    BLANK = auto()
    START = auto()
    END = auto()
    PATH = auto()
    OPEN = auto()
    CLOSED = auto()
    OBSTACLE = auto()


class Node:
    def __init__(self, row, col, node_type: NodeType = NodeType.BLANK):
        self.node_type: NodeType = node_type
        self.row: int = row
        self.col: int = col
        self.parent: Node | None = None
        self.neighbors: List["Node"] = []
        self.g = float("inf")
        self.h = 0

    def change_type(self, new_type: NodeType):
        self.node_type = new_type

    def reset(self, keep_type=False):
        if not keep_type:
            self.node_type = NodeType.BLANK
        self.g = float("inf")
        self.h = 0
        self.parent = None

    def calculate_heuristic(self, node: "Node", method=None):
        x1, y1 = self.row, self.col
        x2, y2 = node.row, node.col
        if method == "manhattan":
            self.h = abs(x1 - x2) + abs(y1 - y2)
        elif method == "chebyshev":
            self.h = max(abs(x2 - x1), abs(y2 - y1))
        else:  # euclidean
            self.h = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return self.h


class Cell:
    type_colors = {
        NodeType.BLANK: Colors.L_GRAY,
        NodeType.START: Colors.ORANGE,
        NodeType.END: Colors.TURQUOISE,
        NodeType.PATH: Colors.YELLOW,
        NodeType.OPEN: Colors.D_GREEN,
        NodeType.CLOSED: Colors.RED,
        NodeType.OBSTACLE: Colors.BLACK,
    }

    def __init__(self, node: Node, width):
        self.width = width
        self.node = node
        self.color = self.type_colors[node.node_type]
        self.x = self.node.row * self.width
        self.y = self.node.col * self.width

    def change_type(self, new_type: NodeType):
        self.node.change_type(new_type)
        self.color = self.type_colors[new_type]

    def reset(self, keep_type=False):
        self.node.reset(keep_type)
        self.color = self.type_colors[self.node.node_type]

    @property
    def get_color(self):
        return self.type_colors[self.node.node_type]

    def draw_cell(self, window):
        pygame.draw.rect(
            window, self.get_color, (self.x, self.y, self.width, self.width)
        )


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid: List[List[Node]] = []

    def make_grid(self):
        for i in range(self.rows):
            self.grid.append([])
            for j in range(self.cols):
                node = Node(i, j)
                self.grid[i].append(node)


class GameGrid:
    # We`ll be assuming a square grid with square cells because I said so
    def __init__(self, rows, window_size, window):
        self.rows = rows
        self.logic_grid: Grid = Grid(self.rows, self.rows)
        self.cell_grid: List[List[Cell]] = []
        self.window_size = window_size
        self.window = window
        self.cell_width = window_size // self.rows
        self.draw_info = None
        self.draw_lines = None

    def get_clicked_cell(self, click_position):
        y, x = click_position
        row = y // self.cell_width
        col = x // self.cell_width
        clicked_cell = self.cell_grid[row][col]
        return clicked_cell

    def draw_grid_lines(self):
        for i in range(self.rows):
            pygame.draw.line(
                self.window,
                Colors.GREY,
                (0, i * self.cell_width),
                (self.window_size, i * self.cell_width),
            )
            pygame.draw.line(
                self.window,
                Colors.GREY,
                (i * self.cell_width, 0),
                (i * self.cell_width, self.window_size),
            )

    def make_cell_grid(self):
        self.logic_grid.make_grid()
        for i, row in enumerate(self.logic_grid.grid):
            self.cell_grid.append([])
            for j, node in enumerate(row):
                cell = Cell(node, self.cell_width)
                self.cell_grid[i].append(cell)

    def reset(self, clear_user_input=True):
        if clear_user_input:
            for i in range(self.rows):
                for j in range(self.rows):
                    self.cell_grid[i][j].reset()
        else:
            for i in range(self.rows):
                for j in range(self.rows):
                    if self.cell_grid[i][j].node.node_type not in [
                        NodeType.START,
                        NodeType.END,
                        NodeType.OBSTACLE,
                    ]:
                        self.cell_grid[i][j].reset()
                    else:
                        self.cell_grid[i][j].reset(keep_type=True)

    def set_draw_info(self, func, **params):
        # partial is fun. Sorry to whoever is reading this (unless you are me, then I am not so sorry)
        self.draw_info = partial(func, **params) if func else None

    def set_draw_lines(self, func):
        self.draw_lines = partial(func) if func else None

    def draw(self):
        self.window.fill(Colors.WHITE)
        for row in self.cell_grid:
            for cell in row:
                cell.draw_cell(self.window)
        self.draw_grid_lines()
        if self.draw_info:
            self.draw_info()
        if self.draw_lines:
            self.draw_lines()
        pygame.display.update()


class TSPSolver:
    def __init__(self, cities, method="bruteforce"):
        self.cities = cities
        self.method = method
        self.memo = {}

    def run(self):
        if self.method == "bruteforce":
            return self.solve_bruteforce()

    def solve_bruteforce(self):
        # Brute-force TSP solver (not efficient for large datasets)
        min_distance = float('inf')
        optimal_route = None
        for route in permutations(self.cities):
            total_route_distance = 0
            for i, cell in enumerate(route):
                if i < len(route) - 1:
                    if self.memo.get(f"{cell.x}-{cell.y}-{route[i+1].x}-{route[i+1].y}"):
                        distance = self.memo[f"{cell.x}-{cell.y}-{route[i+1].x}-{route[i+1].y}"]
                    else:
                        distance = cell.node.calculate_heuristic(route[i+1].node)
                        self.memo[f"{cell.x}-{cell.y}-{route[i+1].x}-{route[i+1].y}"] = distance
                        self.memo[f"{route[i+1].x}-{route[i+1].y}-{cell.x}-{cell.y}"] = distance
                    total_route_distance += distance
                else:
                    if self.memo.get(f"{cell.x}-{cell.y}-{route[0].x}-{route[0].y}"):
                        distance = self.memo[f"{cell.x}-{cell.y}-{route[0].x}-{route[0].y}"]
                    else:
                        distance = cell.node.calculate_heuristic(route[0].node)
                        self.memo[f"{cell.x}-{cell.y}-{route[0].x}-{route[0].y}"] = distance
                        self.memo[f"{route[0].x}-{route[0].y}-{cell.x}-{cell.y}"] = distance
                    total_route_distance += distance

            if total_route_distance < min_distance:
                min_distance = total_route_distance
                optimal_route = route

        return time.time(), list(optimal_route), min_distance


class Game:
    def __init__(self, window, window_size, rows, max_cities=9, method=None):
        self.window = window
        self.window_size = window_size
        self.rows = rows
        self.max_cities = max_cities
        self.method = method
        self.total_time = 0
        self.min_cost = 0
        self.path = []
        self.tsp = None
        self.cities: List[Cell] = []
        self.running = True
        self.game_grid = GameGrid(rows, window_size, window)

    def draw_info(self, method, total_time, cost):
        info_text = f"Method: {str.capitalize(method)} - Cost: {cost:.3f} - Execution time: {total_time:.3f}s"
        font = pygame.font.Font(
            pygame.font.get_default_font(),
            ((self.window_size * 2) - 10) // len(info_text),
        )
        text_surface = font.render(info_text, True, Colors.BLACK)
        self.window.blit(text_surface, (10, 10))

    def update_info(self):
        if self.game_grid.draw_info:
            self.game_grid.set_draw_info(
                self.draw_info,
                method=self.method,
                cost=self.min_cost,
                total_time=self.total_time,
            )

    def setup(self):
        self.game_grid.make_cell_grid()
        self.game_grid.set_draw_info(
            self.draw_info,
            method=self.method,
            cost=self.min_cost,
            total_time=self.total_time,
        )
        self.game_grid.set_draw_lines(self.draw_lines)

    def draw_lines(self):
        path = self.path
        offset = self.game_grid.cell_width / 2
        for i, cell in enumerate(path):
            if i < len(path) - 1:
                pygame.draw.line(WIN, Colors.YELLOW, (cell.x + offset, cell.y + offset),
                                 (path[i + 1].x + offset, path[i + 1].y + offset), 3)
            else:
                pygame.draw.line(WIN, Colors.YELLOW, (cell.x + offset, cell.y + offset),
                                 (path[0].x + offset, path[0].y + offset), 3)

    def run(self):
        self.setup()
        while self.running:
            self.game_grid.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if pygame.mouse.get_pressed()[0]:  # left click
                    self.path = []
                    click_pos = pygame.mouse.get_pos()
                    clicked_cell = self.game_grid.get_clicked_cell(click_pos)
                    if clicked_cell not in self.cities:
                        if len(self.cities) >= self.max_cities:
                            popped_cell = self.cities.pop(0)
                            popped_cell.change_type(NodeType.BLANK)
                        self.cities.append(clicked_cell)
                        clicked_cell.node.change_type(NodeType.START)

                elif pygame.mouse.get_pressed()[2]:  # right click
                    self.path = []
                    click_pos = pygame.mouse.get_pos()
                    clicked_cell = self.game_grid.get_clicked_cell(click_pos)
                    clicked_cell.reset()
                    if clicked_cell in self.cities:
                        self.cities.remove(clicked_cell)

                # start TSPSolver
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and len(self.cities) > 1:
                        self.tsp = TSPSolver(cities=self.cities, method=self.method)
                        self.path = []
                        self.game_grid.draw()
                        time_before = time.time()
                        done, path, min_cost = self.tsp.run()
                        self.total_time = done - time_before
                        self.min_cost = min_cost
                        self.path = path
                        self.update_info()
                        print(f"Execution took {self.total_time:.3f}s")
                    if event.key == pygame.K_c:  # Reset grid
                        self.game_grid.reset()
                        self.cities = []
                        self.path = []
                    if event.key == pygame.K_h:  # Change heuristic
                        self.update_info()
                        print(f"Method updated to: {self.method}")
                    if event.key == pygame.K_i:
                        if self.game_grid:
                            # injects a function (like a callback?) with the parameters already set
                            # just so the drawing is done in one place only. If I draw it here the info would
                            # disappear while the algorithm is running... Maybe I just don`t know how to do things
                            # good ol friend tunnel vison
                            self.game_grid.set_draw_info(
                                self.draw_info,
                                method=self.method,
                                cost=self.min_cost,
                                total_time=self.total_time,
                            ) if not self.game_grid.draw_info else self.game_grid.set_draw_info(
                                None
                            )

        pygame.quit()


if __name__ == "__main__":
    WINDOW_WIDTH = 800
    WIN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_WIDTH))
    pygame.display.set_caption("TSP Playground")
    ROWS = 40
    MAX_CITIES = 9  # gets VERY slow after 9 cities (since it's bruteforcing for now)
    METHOD = "bruteforce"
    new_game = Game(WIN, WIN.get_width(), ROWS, max_cities=MAX_CITIES, method=METHOD)
    pygame.font.init()
    new_game.run()
