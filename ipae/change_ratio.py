import os

ratio = 3
num_of_processes = 50
dir_input = "15-puzzle/15_puzzle_inputs/100_" + str(num_of_processes) + "_1"
dir_output = "15-puzzle/15_puzzle_inputs/100_" + str(num_of_processes) + "_" + str(ratio)
# dir_input = "15-puzzle/15_puzzle_inputs/L100_" + str(num_of_processes) + "_1"
# dir_output = "15-puzzle/15_puzzle_inputs/L100_" + str(num_of_processes) + "_" + str(ratio)
# dir_input = "15-puzzle/15_puzzle_inputs/helper_folder"
# dir_output = "15-puzzle/15_puzzle_inputs/helper_folder2"
original_inputs = os.listdir(dir_input)
for input in original_inputs:
    input_file = open(dir_input + "/" + input, 'r')
    output_file = open(dir_output + "/" + input, 'w')
    try:
        output_file.write("base-level actions:\t[({}, 1000000), ({}, 1000000), ({}, 1000000), ({}, 1000000)]\n".format(ratio, ratio, ratio, ratio))
        lines = input_file.readlines()
        for line in lines[1:]:
            output_file.write(line)
    finally:
        input_file.close()
        output_file.close()