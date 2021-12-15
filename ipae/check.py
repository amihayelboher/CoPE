# import math
#
# def standard_deviation(ratio_of_success):
#     successes = 100 * ratio_of_success
#     failures = 100 - successes
#     std_sum = ((1 - ratio_of_success) ** 2) * successes + (ratio_of_success ** 2) * failures
#     std_div = std_sum / 100
#     return math.sqrt(std_div)
#
# for i in range(100):
#     print(i, standard_deviation(i / 100))



# from multiprocessing import Process
# import time
#
# def do_actions():
#     """
#     Function that should timeout after 5 seconds. It simply prints a number and waits 1 second.
#     :return:
#     """
#     i = 0
#     while True:
#         i += 1
#         print(i)
#         time.sleep(1)
#
#
# if __name__ == '__main__':
#     # We create a Process
#     action_process = Process(target=do_actions)
#
#     # We start the process and we block for 5 seconds.
#     action_process.start()
#     action_process.join(timeout=3)
#
#     # We terminate the process.
#     action_process.terminate()
#     print("Hey there! I timed out! You can do things after me!")


# import re
#
# path_length = {}
# problem_file = open("RCLL_JSONs/problem-001-r1-o1-durations.pddl", 'r')
# for line in problem_file.readlines():
#     if line.find("(= (path-length ") > -1:
#         line = line.strip()[len("(= (path-length "):-1]
#         params = line[:line.find(") ")]
#         # duration = line[line.find(") ") + 2:]
#         duration = re.findall("\d+\.\d+", line)[0]
#         path_length[params.lower()] = duration
# problem_file.close()
# durations = {}
# domain_file = open("RCLL_JSONs/rcll_domain_production_durations_time_windows.pddl", 'r')
# for line in domain_file.readlines():
#     index = line.find("(:action ")
#     if index > line.find(";") and line.find(";") != -1:
#         index = -1
#     if index != -1:
#         durations[line[index + len("(:action "):-1]] = 0
# print()
# domain_file.seek(0)
# found_durative_action = False
# lines = domain_file.readlines()
# for i, line in enumerate(lines):
#     index = line.find("(:durative-action ")
#     if index > line.find(";") and line.find(";") != -1:
#         index = -1
#     if index != -1:
#         action = line[index + len("(:durative-action "):-1]
#         found_durative_action = True
#     if found_durative_action:
#         l = lines[i+1]
#         j = 2
#         index = l.find(":duration (= ?duration ")
#         while index == -1 or (l.find(";") > -1 and index > l.find(";")):
#             l = lines[i+j]
#             j += 1
#             index = l.find(":duration (= ?duration ")
#         durations[action] = l.strip()[len(":duration (= ?duration "):-1]
#         if durations[action].find("(path-length ") > -1:
#             durations[action] = "path-length"
#             found_durative_action = False
#             continue
#         durations[action] = float(durations[action])
#     found_durative_action = False
# domain_file.close()