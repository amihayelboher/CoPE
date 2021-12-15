import json
import math
import re
import heapq
from collections import deque
from copy import deepcopy


# json_file_name = 'searchtree0.json'
json_file_name = 'rcll_001_r1.json'
# json_file_name = 'temp.json'
domain_file_name = "rcll_domain_production_durations_time_windows.pddl"
problem_file_name = "problem-001-r1-o1-durations.pddl"


path_length = {}
problem_file = open("RCLL_JSONs/" + problem_file_name, 'r')
for line in problem_file.readlines():
    if line.find("(= (path-length ") > -1:
        line = line.strip()[len("(= (path-length "):-1]
        params = line[:line.find(") ")]
        # duration = line[line.find(") ") + 2:]
        duration = re.findall("\d+\.\d+", line)[0]
        path_length[params.lower()] = duration
problem_file.close()

durations = {}
domain_file = open("RCLL_JSONs/" + domain_file_name, 'r')
for line in domain_file.readlines():
    index = line.find("(:action ")
    if index > line.find(";") and line.find(";") != -1:
        index = -1
    if index != -1:
        durations[line[index + len("(:action "):-1]] = 0
domain_file.seek(0)
lines = domain_file.readlines()
for i, line in enumerate(lines):
    index = line.find("(:durative-action ")
    if index > line.find(";") and line.find(";") != -1:
        index = -1
    if index != -1:
        action = line[index + len("(:durative-action "):-1]
        l = lines[i+1]
        j = 2
        index = l.find(":duration (= ?duration ")
        while index == -1 or (l.find(";") > -1 and index > l.find(";")):
            l = lines[i+j]
            j += 1
            index = l.find(":duration (= ?duration ")
        durations[action] = l.strip()[len(":duration (= ?duration "):-1]
        if durations[action].find("(path-length ") > -1:
            durations[action] = "path-length"
            continue
        durations[action] = float(durations[action])
domain_file.close()

blas_indices = {"TIL": 0, None: ""}
i = 1
for k in durations.keys():
    blas_indices[k] = str(i)
    i += 1
for k in path_length.keys():
    blas_indices[k] = str(i)
    i += 1

dur_by_index = {"0": 0}
for k in durations.keys():
    dur_by_index[blas_indices[k]] = durations[k] if durations[k] != "path-length" else 0
for k in path_length.keys():
    dur_by_index[blas_indices[k]] = path_length[k]
blas_and_durs = []
for i in range(len(dur_by_index.keys())):
    blas_and_durs.append((i, dur_by_index[str(i)]))


with open('RCLL_JSONs/' + json_file_name) as json_file:
    json_dict = json.loads(json_file.read())
expansions_per_second, nodes_list = int(json_dict["expansionsPerSecond"]), json_dict["nodes"]

nodes_dict = {} # dictionary of the nodes using the id field as key
already_treated = {}
for node in nodes_list:
    nodes_dict[node['id']] = node
    already_treated[node['id']] = False
    action = None
    duration = 0
    if 'reachedBy' in node.keys():
        action = node['reachedBy'][0]
        if len(node['reachedBy']) > 1 and action.find("TIL") > -1:
            action = node['reachedBy'][1]
        if action.find('(') > -1:
            action = action[1 : action.find(')')]
        if action.find("TIL") > -1:
            action = "TIL"
        else:
            action_name = action
            if action_name.find(' ') > -1:
                action_name = action_name[:action_name.find(' ')]
            duration = durations[action_name]
            if duration == "path-length":
                params = action[action.find(' ') + 1:]
                params_without_agent = params[params.find(' ') + 1:].lower()
                if params_without_agent.count(' ') == 2:
                    params_without_agent += " input"
                duration = math.ceil(float(path_length[params_without_agent]))
                action = params_without_agent
            else:
                action = action_name
    node['action'] = action
    node['duration'] = duration
    node['BLA'] = blas_indices[action]
    node['H'] = []
    node['depth'] = 0
    node['h*'] = math.inf
    node['closest_goal_id'] = None
    if 'duplicates' in node.keys():
        for dup_id in node['duplicates']:
            nodes_dict[dup_id] = node
            already_treated[dup_id] = False

tree_nodes_queue = deque()
root = nodes_dict["0"]
root['parent'] = None
tree_nodes_queue.append(root)
while tree_nodes_queue:
    cur_node = tree_nodes_queue.popleft()
    if 'duplicates' in cur_node.keys():
        for dup_id in cur_node['duplicates']:
            already_treated[dup_id] = True
    cur_node['existSuccessors'] = []
    for suc_id in cur_node['successors']:
        if not already_treated[suc_id]:
            suc_node = nodes_dict[suc_id]
            tree_nodes_queue.append(suc_node)
            already_treated[suc_node['id']] = True
            cur_node['existSuccessors'].append(suc_id)
            suc_node['H'] = deepcopy(cur_node['H'])
            suc_node['H'].append(suc_node['BLA'])
            suc_node['depth'] = cur_node['depth'] + 1
            suc_node['h*'], suc_node['closest_goal_id'] = (0, suc_id) if suc_node['tag'] == "goal" else (math.inf, None)
            ancestor_node = cur_node
            while suc_node['h*'] < math.inf and ancestor_node['h*'] > suc_node['depth'] - ancestor_node['depth']:
                ancestor_node['h*'] = suc_node['depth'] - ancestor_node['depth']
                ancestor_node['closest_goal_id'] = suc_id
                if ancestor_node['parent']:
                    ancestor_node = nodes_dict[ancestor_node['parent']]
                else:
                    break


#####################################################################################################
# if the version of the planner doesn't compute the expansionDelay field, use the next code snippet #
# (due to the condition in the beginning, no special commenting or uncommenting is needed)          #
#####################################################################################################

if not 'expansionDelay' in nodes_dict['0'].keys():
    was_already_inserted = {}
    for node in nodes_list:
        was_already_inserted[node['id']] = False
        if 'duplicates' in node.keys():
            for dup_id in node['duplicates']:
                was_already_inserted[dup_id] = False
    root = nodes_dict['0']
    open_list = [(root['distanceToGo'], 0, 0, root)]
    was_already_inserted['0'] = True
    expansions_counter = 1
    insertions_counter = 0
    while open_list:
        (_, parentExpansionIndex, _, n) = heapq.heappop(open_list)
        n['expansionIndex'] = expansions_counter
        n['expansionDelay'] = parentExpansionIndex + parentExpansionIndex
        expansions_counter += 1
        for child_id in n['successors']:
            child = nodes_dict[child_id]
            if child['tag'] != 'frontier' and not was_already_inserted[child_id]:
                was_already_inserted[child_id] = True
                if 'duplicates' in child.keys():
                    for dup_id in child['duplicates']:
                        was_already_inserted[dup_id] = True
                child_triplet = (child['depth'] + child['distanceToGo'], n['expansionIndex'], insertions_counter, child)
                insertions_counter += 1
                # print(child_id, child_triplet)
                heapq.heappush(open_list, child_triplet)

# if not 'expansionDelay' in nodes_dict['0'].keys():
#     root = nodes_dict['0']
#     open_list = Heap.MinHeap(len(nodes_dict))
#     open_list.insert((root['distanceToGo'], 0, 0, root))
#     expansions_counter = 1
#     insertions_counter = 0
#     while open_list:
#         (_, parentExpansionIndex, _, n) = open_list.extract_min()
#         n['expansionIndex'] = expansions_counter
#         n['expansionDelay'] = parentExpansionIndex + parentExpansionIndex
#         expansions_counter += 1
#         for child_id in n['successors']:
#             child = nodes_dict[child_id]
#             if child['tag'] != 'frontier':
#                 child_triplet = (child['depth'] + child['distanceToGo'], n['expansionIndex'], insertions_counter, child)
#                 insertions_counter += 1
#                 # print(child_id, child_triplet)
#                 open_list.insert(child_triplet)

#####################################################################################################
# until here                                                                                        #
#####################################################################################################


avg_expansion_delay = 0
counted_nodes = 0
avg_expansion_delay_list = [0] * 200
counted_nodes_list = [0] * 200
for node in nodes_list:
    if 'expansionDelay' in node.keys():
        avg_expansion_delay += node['expansionDelay']
        counted_nodes += 1
        if node['depth'] < 200:
            avg_expansion_delay_list[node['depth']] += node['expansionDelay']
            counted_nodes_list[node['depth']] += 1
avg_expansion_delay /= counted_nodes
# print("avg_expansion_delay:", avg_expansion_delay)
for i in range(len(avg_expansion_delay_list)):
    if counted_nodes_list[i] != 0:
        avg_expansion_delay_list[i] /= counted_nodes_list[i]
# print("avg_expansion_delay_list:", avg_expansion_delay_list[:10])


nodes_by_expansion_index = {}
for n in nodes_list:
    if n['tag'] != "frontier" and was_already_inserted[n['id']]:
        nodes_by_expansion_index[n['expansionIndex']] = n
file_data_extracted = open("RCLL_JSONs/rcll_data_extracted_file" + ".txt", 'w')
file_data_extracted.write("BLAs: " + str(blas_and_durs) + "\n")
insertion_keys = nodes_by_expansion_index.keys()
insertion_order = [key for key in insertion_keys]
insertion_order.sort()
# for node in nodes_list:
for i in insertion_order:
    node = nodes_by_expansion_index[i]
    dtg = node['distanceToGo'] if type(node['distanceToGo']) == int else '-'
    num_of_expanded_simple = math.ceil((node['h*'] + 1) * avg_expansion_delay) if node['h*'] != math.inf else math.inf
    num_of_expanded_complex = node['h*']
    if num_of_expanded_complex != math.inf:
        num_of_expanded_complex = 0
        for j in range(node['h*'] + 1):
            num_of_expanded_complex += avg_expansion_delay_list[node['depth'] + j]
        num_of_expanded_complex = math.ceil(num_of_expanded_complex)
    ctg = node['costToGo'] if node['costToGo'] < 1000000 else '-'
    str_to_write = "h: " + str(dtg) + ",\t" +\
                   "h*: " + str(node['h*']) + ",\t" +\
                   "m1: " + str(num_of_expanded_simple) + ",\t" + \
                   "m2: " + str(num_of_expanded_complex) + ",\t" + \
                   "c: " + str(ctg) + ",\t" + \
                   "BLA: " + str(node['BLA']) + ",\t" + \
                   "id: " + node['id'] + ",\t" + \
                   "H: " + str(node['H']) + "\n"
    # if num_of_expanded_simple == math.inf:
    #     continue
    file_data_extracted.write(str_to_write)
file_data_extracted.close()