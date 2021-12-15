import random
import time
from heapq import heappush, heappop
from copy import deepcopy

def manhattan_distance(puzzle, goals_locations):
    distance = 0
    for i in range(4):
        for j in range(4):
            if puzzle[i][j] != 0:
                g_i, g_j = goals_locations[puzzle[i][j]]
                distance += abs(i - g_i) + abs(j - g_j)
    return distance

def find_zero_indices(puzzle):
    i, j = 0, 0
    for i in range(4):
        for j in range(4):
            if puzzle[i][j] == 0:
                return i, j

def swap(puzzle, i1, j1, i2, j2):
    swapped = deepcopy(puzzle)
    swapped[i1][j1], swapped[i2][j2] = swapped[i2][j2], swapped[i1][j1]
    return swapped

def expand(min_element, goals_locations):
    children = []
    i, j = find_zero_indices(min_element[-1])
    if i > 0:
        swapped = swap(min_element[-1], i, j, i - 1, j)
        children.append([min_element[2] + manhattan_distance(swapped, goals_locations), min_element, min_element[2] + 1, swapped])
    if i < 3:
        swapped = swap(min_element[-1], i, j, i + 1, j)
        children.append([min_element[2] + manhattan_distance(swapped, goals_locations), min_element, min_element[2] + 1, swapped])
    if j > 0:
        swapped = swap(min_element[-1], i, j, i, j - 1)
        children.append([min_element[2] + manhattan_distance(swapped, goals_locations), min_element, min_element[2] + 1, swapped])
    if j < 3:
        swapped = swap(min_element[-1], i, j, i, j + 1)
        children.append([min_element[2] + manhattan_distance(swapped, goals_locations), min_element, min_element[2] + 1, swapped])
    return children

def create_goal_puzzle_and_locations_dict():
    goal_puzzle = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    goals_locations = {}
    for i in range(4):
        for j in range(4):
            goals_locations[goal_puzzle[i][j]] = (i, j)
    return goal_puzzle, goals_locations

def solve_puzzle(initial_puzzle, fringe_size=0, relative=False):
    # solve the puzzle using A-star algorithm
    goal_puzzle, goals_locations = create_goal_puzzle_and_locations_dict()
    min_heap = []
    initial_h_value = manhattan_distance(initial_puzzle, goals_locations)
    heappush(min_heap, [initial_h_value, None, 0, initial_puzzle])
    expanded = set()
    expansions = 0
    beginning_time = time.time()
    while min_heap:
        # if expansions % 1000 == 0 and expansions != 0:
        #     print("expansions: " + str(expansions) + ", time: " + str(time.time() - beginning_time) + ", ratio: " + str(expansions / (time.time() - beginning_time)))
        if fringe_size > 0 and len(min_heap) >= fringe_size:
            if relative:
                return initial_h_value, expansions, min_heap[:fringe_size]
            return int(10000 * (time.time() - beginning_time)), min_heap[:fringe_size]
        min_element = heappop(min_heap)
        if min_element[-1] == goal_puzzle:
            return expansions, min_element
        str_min_element = str(min_element[-1])
        if str_min_element not in expanded:
            expansions += 1
            expanded.add(str_min_element)
            for child_element in expand(min_element, goals_locations):
                heappush(min_heap, child_element)
    return expansions, []

def print_solution(last_element):
    print("solution:\n")
    solution_length = last_element[2] + 1
    path = [None for _ in range(solution_length)]
    current_element = last_element
    step_number = solution_length - 1
    while current_element:
        path[step_number] = current_element[-1]
        current_element = current_element[1]
        step_number -= 1
    for state in path:
        for line in state:
            print(line)
        print()

def count_inversions(num_list):
    inversions = 0
    for i in range(len(num_list)):
        for j in range(i + 1, len(num_list)):
            if num_list[i] > num_list[j]:
                inversions += 1
    return inversions

def is_solvable(puzzle):
    i = 1
    for line in puzzle:
        if i in line:
            break
        i += 1
    position_as_line = [num for line in puzzle for num in line]
    inversions = count_inversions(position_as_line)
    return True if (i + inversions) % 2 == 1 else False

def create_puzzle():
    perm = random.sample(range(16), 16)
    puzzle = [[perm[4*i], perm[4*i+1], perm[4*i+2], perm[4*i+3]] for i in range(4)]
    while not is_solvable(puzzle):
        perm = random.sample(range(16), 16)
        puzzle = [[perm[4*i], perm[4*i+1], perm[4*i+2], perm[4*i+3]] for i in range(4)]
    return puzzle

def create_puzzle2(shuffling_steps=50):
    puzzle, goals_locations = create_goal_puzzle_and_locations_dict()
    for _ in range(shuffling_steps):
        optional_next_puzzles = [element[-1] for element in expand([0, None, 0, puzzle], goals_locations)]
        puzzle = optional_next_puzzles[random.randint(0, len(optional_next_puzzles) - 1)]
    return puzzle


def play(print_initial_state=True, print_manhattan_distance=True, print_expansions=True, print_solution_length=True,
                print_solution_path=False, collect_data=True, fringe_size=0, relative=False):
    _, goals_locations = create_goal_puzzle_and_locations_dict()
    if fringe_size > 0:
        fringe = []
        while len(fringe) < fringe_size:
            random_puzzle = create_puzzle2(30)
            dist = manhattan_distance(random_puzzle, goals_locations)
            # if dist <= 6:
            if dist <= 16:
                continue
            if relative:
                ret = solve_puzzle(random_puzzle, fringe_size, relative=relative)
                if len(ret) == 2:
                    fringe = []
                    continue
                init_h, _, fringe = ret
            else:
                _, fringe = solve_puzzle(random_puzzle, fringe_size)
        if relative:
            return init_h, fringe
        else:
            return fringe
    # random_puzzle = create_puzzle()
    # collect_data = False
    random_puzzle = create_puzzle2(50)
    if print_initial_state:
        print("initial state:", random_puzzle)
    if print_manhattan_distance:
        print("manhattan distance:", manhattan_distance(random_puzzle, goals_locations))
    else:
        expansions, last_element = solve_puzzle(random_puzzle)
    if print_expansions:
        print("expansions:", expansions)
    if print_solution_length:
        print("solution length:", last_element[2] + 1)
    if print_solution_path:
        print_solution(last_element)
    if collect_data:
        file = open("games_data_part2.txt", 'a')
        try:
            file.write(str(random_puzzle) + "\t" + str(last_element[2]) + "\t" + str(expansions) + "\n")
        finally:
            file.close()


if __name__ == '__main__':
    try:
        games_played = 0
        while games_played < 1000:
            if games_played % 500 == 0:
                print(games_played)
            play(print_initial_state=False, print_manhattan_distance=False, print_expansions=False, print_solution_length=False,
                print_solution_path=False, collect_data=True)
            games_played += 1
    except KeyboardInterrupt:
        pass