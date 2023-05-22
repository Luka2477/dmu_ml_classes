import random
import numpy as np
import matplotlib.pyplot as plt


class Pos:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __add__(self, other):
        return Pos(self.row + other.row, self.col + other.col)

    def __mul__(self, other):
        return Pos(self.row * other.row, self.col * other.col)

    def __str__(self):
        return f"({self.row}, {self.col})"

    def invert(self):
        self.row *= -1
        self.col *= -1
        return self

    @staticmethod
    def random(a: int, b: int) -> "Pos":
        return Pos(random.randint(a, b), random.randint(a, b))


def plot_map(inner_map):
    colors = ["white", "magenta", "red", "red", "blue"]
    markers = ["", "x", "v", "^", "1"]
    for row in range(len(inner_map)):
        for col in range(len(inner_map[0])):
            plt.scatter(col, abs(len(inner_map) - 1 - row), c=colors[inner_map[row][col]],
                        marker=markers[inner_map[row][col]])


# Exercise 1:
# Make a construction of the "map", holding information about the obstacles, start and so on.
#
real_spaces = dict(open=0, obstacle=1, start=2, finish=3)
real_map = \
    [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 10
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],  # 9
     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # 8
     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 7
     [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # 6
     [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 5
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # 4
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
     [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],  # 2
     [0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 1
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]  # 0
#     0  1  2  3  4  5  6  7  8  9  10
real_start = Pos(9, 1)
real_finish = Pos(1, 9)

# Show map
plot_map(real_map)
plt.title("Original map")
plt.show()


# Exercise 2:
# Write in pseudocode an algorithm that makes the robot explore the most in 50 time steps
#
# Initialize robot map with all fields being -1 simulation an obstacle
# While time steps left is not 0 do
#     Update robot map with visited robot pos
#     Check all spaces around robot pos
#     Update robot map with checked spaces
#     If robot pos is equal to finish pos
#         Find the shortest route
#         Jump out of while loop
#     Else if there exists an open space diagonally to robot pos do
#         Move robot to open space
#     Else if there exists an open space adjacent to robot pos do
#         Move robot to open space
#     Else do
#         If there exists a visited space diagonally to robot pos do
#             Move robot to visited space
#         Else if there exists a visited space adjacent to robot pos do
#             Move robot to visited space
#         Else do
#             Throw invalid mapping error
#             Jump out of while loop

# Exercise 4:
# Implement the robot
#
class Robot:
    def __init__(self):
        # self.spaces = dict(open=0, obstacle=1, start=2, finish=3, visited=4, route_visited=5, route=6)
        self.spaces = dict(open=0, obstacle=1, start=2, finish=3, visited=4, route=5)
        self.map = [[self.spaces["obstacle"] for _ in range(len(real_map[0]))] for _ in range(len(real_map))]
        self.map[real_start.row][real_start.col] = self.spaces["start"]
        self.map[real_finish.row][real_finish.col] = self.spaces["finish"]
        self.start = real_start
        self.finish = None
        self.pos = self.start
        self.last_pos = []
        self.should_use_desired_direction = True
        self.desired_direction = Pos(-1, 1)

    # Check if the position is within the bounds of the map
    @staticmethod
    def is_valid_pos(pos: Pos) -> bool:
        return 0 <= pos.row < len(real_map) and 0 <= pos.col < len(real_map[0])

    # Check if the position is valid and if it is not an obstacle or already visited
    def is_valid_move(self, pos: Pos) -> bool:
        return self.is_valid_pos(pos) \
            and self.map[pos.row][pos.col] not in [self.spaces["obstacle"], self.spaces["visited"]]

    # Check if the robot is on the finish
    def is_finish(self):
        if self.pos == real_finish:
            self.finish = self.pos
            return True
        return False

    # Check if any moves are valid
    def has_valid_move(self):
        for row in range(-1, 2):
            for col in range(-1, 2):
                if self.is_valid_move(self.pos + Pos(row, col)):
                    return True
        return False

    # Check if the position is in one of the corners
    def has_hit_corner(self):
        return (self.pos.row == 0 and (self.pos.col == 0 or self.pos.col == len(self.map[0]) - 1)) \
            or (self.pos.row == len(self.map) - 1 and (self.pos.col == 0 or self.pos.col == len(self.map[0]) - 1))

    # Update map in all directions if real_map value is open or obstacle
    def look_around(self) -> None:
        for row in range(-1, 2):
            for col in range(-1, 2):
                new_row, new_col = self.pos.row + row, self.pos.col + col
                if self.is_valid_pos(Pos(new_row, new_col)):
                    if real_map[new_row][new_col] in [real_spaces["open"], real_spaces["obstacle"]] \
                            and self.map[new_row][new_col] != self.spaces["visited"]:
                        self.map[new_row][new_col] = real_map[new_row][new_col]
                    elif self.map[new_row][new_col] == self.spaces["finish"]:
                        self.finish = Pos(new_row, new_col)

    # Move to the desired directio or random depending on what is valid and if any are valid
    def move(self) -> None:
        if self.has_hit_corner():
            self.desired_direction.invert()

        if not self.has_valid_move():
            self.pos = self.last_pos.pop()
        elif self.should_use_desired_direction and self.is_valid_move(self.pos + self.desired_direction):
            self.last_pos.append(self.pos)
            self.pos += self.desired_direction
        else:
            valid_move_found = False
            while not valid_move_found:
                new_pos = self.pos + Pos.random(-1, 1)
                if self.is_valid_move(new_pos):
                    self.last_pos.append(self.pos)
                    self.pos = new_pos
                    valid_move_found = True

        if self.map[self.pos.row][self.pos.col] == self.spaces["open"]:
            self.map[self.pos.row][self.pos.col] = self.spaces["visited"]

    # Helper method to neutralize the map before use for Dijkstras al.
    def dijkstras_map_func(self, col: int):
        if col != self.spaces["obstacle"]:
            return 1
        return np.inf

    # Backtracking function
    # Idea start at the last node then choose the least number of steps to go back
    def backtrack(self, start: Pos, finish: Pos, distances: list[list[int]]):
        path = [finish]

        while True:
            potential_distances = []
            potential_positions = []

            for drow in range(-1, 2):
                for dcol in range(-1, 2):
                    potential_pos = path[-1] + Pos(drow, dcol)
                    if self.is_valid_pos(potential_pos):
                        potential_positions.append(potential_pos)
                        potential_distances.append(distances[potential_pos.row][potential_pos.col])

            shortest_distance_index = np.argmin(potential_distances)
            path.append(potential_positions[shortest_distance_index])

            if path[-1] == start:
                break

        return path

    # Dijkstras algorithm for finding the shortest path between two nodes in a graph.
    def dijkstras(self, start: Pos, finish: Pos):
        path_map = [[self.dijkstras_map_func(col) for col in row] for row in self.map]
        path_map[start.row][start.col] = 0.0
        path_map[finish.row][finish.col] = 0.0

        visited = [[False for _ in range(len(self.map))] for _ in range(len(self.map[0]))]

        distances = [[np.inf for _ in range(len(self.map))] for _ in range(len(self.map[0]))]
        distances[start.row][start.col] = 0

        curr_pos = Pos(start.row, start.col)
        while True:
            for drow in range(-1, 2):
                for dcol in range(-1, 2):
                    potential_pos = curr_pos + Pos(drow, dcol)
                    if self.is_valid_pos(potential_pos) and not visited[potential_pos.row][potential_pos.col]:
                        distance = distances[curr_pos.row][curr_pos.col] + \
                                   path_map[potential_pos.row][potential_pos.col]

                        if distance < distances[potential_pos.row][potential_pos.col]:
                            distances[potential_pos.row][potential_pos.col] = distance

            visited[curr_pos.row][curr_pos.col] = True

            t = [
                [col if not visited[rowi][coli] else np.inf for coli, col in enumerate(row)]
                for rowi, row in enumerate(distances)
            ]
            shortest_path_pos = np.unravel_index(np.argmin(t), np.shape(t))
            curr_pos = Pos(shortest_path_pos[0], shortest_path_pos[1])

            if curr_pos == finish:
                break

        return self.backtrack(start, finish, distances)


if __name__ == '__main__':
    robot = Robot()
    time_steps = 50

    while time_steps > 0 and not robot.finish:
        robot.look_around()
        robot.move()

        time_steps -= 1

    # Show searched area
    plot_map(robot.map)
    plt.title("After exploration")
    plt.show()

    # Find the shortest path
    shortest_path = robot.dijkstras(robot.finish or robot.pos, robot.start)

    # Show the shortest path
    plot_map(robot.map)
    plt.plot([pos.col for pos in shortest_path], [abs(len(real_map) - 1 - pos.row) for pos in shortest_path], "g")
    plt.title("With shortest path")
    plt.show()
