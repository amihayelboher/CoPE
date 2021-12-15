import ast

from tile_puzzle_15_a_star import manhattan_distance
from tile_puzzle_15_a_star import create_goal_puzzle_and_locations_dict


input_file = open("games_data_part2.txt", 'r')
output_file = open("m_lists_part2.txt", 'w')
try:
    _, goal_locations = create_goal_puzzle_and_locations_dict()
    histograms = [[] for _ in range(65)]
    sums = [0 for _ in range(65)]
    line = input_file.readline()
    while line:
        state_str, _, expansions_str = line.split('\t')
        state = ast.literal_eval(state_str)
        expansions = int(expansions_str)
        state_distance = manhattan_distance(state, goal_locations)
        histogram = histograms[state_distance]
        i = -1
        for j in range(len(histogram)):
            if histogram[j][0] == expansions:
                i = j
                break
        if i != -1:
            histogram[i][1] += 1
        else:
            histogram.append([expansions, 1])
        sums[state_distance] += expansions
        line = input_file.readline()
    for i, histogram in enumerate(histograms):
        histogram.sort()
        for j in range(len(histogram)):
            if sums[i] != 0:
                histogram[j][1] = (histogram[j][0] * histogram[j][1]) / sums[i]
    output_file.write("[(1.0, 0)]\n")
    for normalized_histogram in histograms[1:]:
        if normalized_histogram == []:
            output_file.write("[(1.0, 1000000)]\n")
        else:
            to_write = "["
            for pair in normalized_histogram:
                to_write += "(" + str(pair[1]) + ", " + str(pair[0]) + "), "
            output_file.write(to_write[:-2] + "]\n")
finally:
    input_file.close()
    output_file.close()