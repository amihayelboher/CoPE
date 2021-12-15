import math
import random

from ipae_mdp import Process
from ipae_mdp import Distribution

from ipae_new_input_generator import dist_to_string
from ipae_new_input_generator import sample_real_times

class PuzzleSolver:
    def __init__(self, strategy):
        """
        :param strategy: Strategy
        """
        self._strategy = strategy

    def print_performance(self):
        print(f'{self._strategy} - Expanded Nodes: {self._strategy.num_expanded_nodes}')

    def print_solution(self):
        print('Solution:')
        for p in self._strategy.solution:
            print(p)

    def run(self, folder, input_index, num_of_processes):
        return self._strategy.do_algorithm(folder, input_index, num_of_processes)


class Strategy:
    num_expanded_nodes = 0
    solution = None

    def do_algorithm(self, folder, input_index, num_of_processes):
        raise NotImplemented


class AStar(Strategy):
    def __init__(self, initial_puzzle):
        """
        :param initial_puzzle: Puzzle
        """
        self.start = initial_puzzle

    def __str__(self):
        return 'A*'

    @staticmethod
    def _calculate_new_heuristic(move, end_node):
        return move.heuristic_manhattan_distance() - end_node.heuristic_manhattan_distance()

    def do_algorithm(self, folder, input_index, num_of_processes):
        queue = [[self.start.heuristic_manhattan_distance(), self.start]]
        expanded = []
        num_expanded_nodes = 0
        path = None

        while queue:
            if len(queue) > 2 * num_of_processes:
                create_input(folder, queue[:num_of_processes], input_index)
                return True
            
            i = 0
            for j in range(1, len(queue)):
                if queue[i][0] > queue[j][0]:  # minimum
                    i = j


            path = queue[i]
            queue = queue[:i] + queue[i + 1:]
            end_node = path[-1]

            if end_node.position == end_node.PUZZLE_END_POSITION:
                break
            if end_node.position in expanded:
                continue

            for move in end_node.get_moves():
                if move.position in expanded:
                    continue
                new_path = [path[0] + self._calculate_new_heuristic(move, end_node)] + path[1:] + [move]
                queue.append(new_path)
                expanded.append(end_node.position)

            num_expanded_nodes += 1
        
        return False


class Puzzle:
    def __init__(self, position):
        """
        :param position: a list of lists representing the puzzle matrix
        """
        self.position = position
        self.PUZZLE_NUM_ROWS = len(position)
        self.PUZZLE_NUM_COLUMNS = len(position[0])
        self.PUZZLE_END_POSITION = self._generate_end_position()

    def __str__(self):
        """
        Print in console as a matrix
        """
        puzzle_string = '—' * 13 + '\n'
        for i in range(self.PUZZLE_NUM_ROWS):
            for j in range(self.PUZZLE_NUM_COLUMNS):
                puzzle_string += '│{0: >2}'.format(str(self.position[i][j]))
                if j == self.PUZZLE_NUM_COLUMNS - 1:
                    puzzle_string += '│\n'

        puzzle_string += '—' * 13 + '\n'
        return puzzle_string

    def _generate_end_position(self):
        """
        Example end position in 4x4 puzzle
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        """
        end_position = []
        new_row = []

        for i in range(self.PUZZLE_NUM_ROWS * self.PUZZLE_NUM_COLUMNS):
            new_row.append(i)
            if len(new_row) == self.PUZZLE_NUM_COLUMNS:
                end_position.append(new_row)
                new_row = []

        return end_position

    def _swap(self, x1, y1, x2, y2):
        """
        Swap the positions between two elements
        """
        puzzle_copy = [list(row) for row in self.position]  # copy the puzzle
        puzzle_copy[x1][y1], puzzle_copy[x2][y2] = puzzle_copy[x2][y2], puzzle_copy[x1][y1]

        return puzzle_copy

    def _get_coordinates(self, tile, position=None):
        """
        Returns the i, j coordinates for a given tile
        """
        if not position:
            position = self.position

        for i in range(self.PUZZLE_NUM_ROWS):
            for j in range(self.PUZZLE_NUM_COLUMNS):
                if position[i][j] == tile:
                    return i, j

        return RuntimeError('Invalid tile value')

    def get_moves(self):
        """
        Returns a list of all the possible moves
        """
        moves = []
        i, j = self._get_coordinates(0)  # blank space

        if i > 0:
            moves.append(Puzzle(self._swap(i, j, i - 1, j)))  # move up

        if j < self.PUZZLE_NUM_COLUMNS - 1:
            moves.append(Puzzle(self._swap(i, j, i, j + 1)))  # move right

        if j > 0:
            moves.append(Puzzle(self._swap(i, j, i, j - 1)))  # move left

        if i < self.PUZZLE_NUM_ROWS - 1:
            moves.append(Puzzle(self._swap(i, j, i + 1, j)))  # move down

        return moves

    def heuristic_misplaced(self):
        """
        Counts the number of misplaced tiles
        """
        misplaced = 0

        for i in range(self.PUZZLE_NUM_ROWS):
            for j in range(self.PUZZLE_NUM_COLUMNS):
                if self.position[i][j] != self.PUZZLE_END_POSITION[i][j]:
                    misplaced += 1

        return misplaced

    def heuristic_manhattan_distance(self):
        """
        Counts how much is a tile misplaced from the original position
        """
        distance = 0

        for i in range(self.PUZZLE_NUM_ROWS):
            for j in range(self.PUZZLE_NUM_COLUMNS):
                i1, j1 = self._get_coordinates(self.position[i][j], self.PUZZLE_END_POSITION)
                distance += abs(i - i1) + abs(j - j1)

        return distance



def zero_indices(pos):
    for i in range(4):
        for j in range(4):
            if pos[i][j] == 0:
                return i, j

def find_move(pos1, pos2):
    '''up - 0, down - 1, left - 2, right - 3'''
    i1, j1 = zero_indices(pos1)
    i2, j2 = zero_indices(pos2)
    if i1 > i2:
        return 0
    if i1 < i2:
        return 1
    if j1 > j2:
        return 2
    return 3

def create_input(folder, queue, input_index):
    processes = []
    for node in queue:
        t = node[-1].heuristic_misplaced()
        p = 1 / (18 - t)
        m = Distribution([(p, math.floor(math.sqrt(node[-1].heuristic_manhattan_distance()))), (1 - p, 10000)])
        d = Distribution([(1, random.randint(t, 2 * t + math.floor(math.sqrt(len(queue)))))])
        H = [(find_move(node[i-1].position, node[i].position), 10000) for i in range(2, len(node))]
        processes.append(Process(m, d, H))
    blas = [(1, 10000), (1, 10000), (1, 10000), (1, 10000)]
    file = open('15_puzzle_input_bank/' + folder + '/input' + str(input_index) + ".txt", 'w')
    try:
        file.write("base-level actions:\t")
        file.write(str(blas) + "\n\n")
        m_lists = [p.m.dist_list for p in processes]
        d_lists = [p.d.dist_list for p in processes]
        real_termination_times = sample_real_times(m_lists)
        real_deadlines = sample_real_times(d_lists)
        file.write("real termination times:\t" + str(real_termination_times) + "\n")
        file.write("real deadlines:\t" + str(real_deadlines) + "\n\n")
        for i, p in enumerate(processes):
            file.write("p" + str(i) + ":\t\n")
            file.write("m:\t" + dist_to_string(p.m.dist_list) + "\n")
            file.write("d:\t" + dist_to_string(p.d.dist_list) + "\n")
            file.write("H:\t" + str(p.H) + "\n\n")
    finally:
        file.close()


def generate_input(puzzle, folder, input_index, num_of_processes):
    p = PuzzleSolver(AStar(puzzle))
    return p.run(folder, input_index, num_of_processes)




def draw_puzzle():
    perm = random.sample(range(16), 16)
    return Puzzle([[perm[4*i], perm[4*i+1], perm[4*i+2], perm[4*i+3]] for i in range(4)])

num_of_inputs = 1
num_of_processes = 10
folder = str(num_of_processes) + "_procs_kd"
i = 0
while i < num_of_inputs:
    if generate_input(draw_puzzle(), folder, i, num_of_processes):
        i += 1