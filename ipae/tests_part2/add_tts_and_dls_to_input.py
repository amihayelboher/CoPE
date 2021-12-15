import os
from ipae_tests_part_2 import build_input_from_text_file


def sample_and_add_to_input(src_folder, src_input, num_of_samples, ret=False):
    processes, base_level_actions, _, _ = build_input_from_text_file(src_folder + "/" + src_input)
    genuine_file = open(src_folder + "/" + src_input, 'r')
    gen_lines = genuine_file.readlines()
    dest_folder = "15-puzzle-multiple-sampling/helper_folder"
    bla_dur = 1
    dest_input = "{}/{}_{}_{}.txt".format(dest_folder, src_input[:-4], str(len(processes)), bla_dur)
    file_copy = open(dest_input, 'w')
    file_copy.write("base-level actions:\t[({}, 1000000), ({}, 1000000), ({}, 1000000), ({}, 1000000)]\n\n".format(bla_dur, bla_dur, bla_dur, bla_dur))
    termination_times_list = []
    deadlines_list = []
    for i in range(num_of_samples):
        tts_sample = [p.m.sample() for p in processes]
        dls_sample = [p.d.sample() for p in processes]
        file_copy.write("real termination times " + str(i) + ":	" + str(tts_sample) + "\n")
        file_copy.write("real deadlines " + str(i) + ":	" + str(dls_sample) + "\n\n")
        termination_times_list.append(tts_sample)
        deadlines_list.append(dls_sample)
    for i in range(5, len(gen_lines)):
        file_copy.write(gen_lines[i])
    genuine_file.close()
    file_copy.close()
    res_file = open("15-puzzle-multiple-sampling/results_{}/{}_{}_{}.txt".format(len(processes), src_input[:-4], str(len(processes)),bla_dur), 'w')
    res_file.close()
    for bla_dur in [2, 3]: # 1 has already been done above and used here as the source to be copied
        from_file = open(dest_input, 'r')
        to_file = open("{}{}.txt".format(dest_input[:-5], bla_dur) ,'w')
        to_file.write("base-level actions:\t[({}, 1000000), ({}, 1000000), ({}, 1000000), ({}, 1000000)]\n".format(bla_dur, bla_dur, bla_dur, bla_dur))
        for line in from_file.readlines()[1:]:
            to_file.write(line)
        res_file = open("15-puzzle-multiple-sampling/results_{}/{}_{}_{}.txt".format(len(processes), src_input[:-4], str(len(processes)), bla_dur), 'w')
        res_file.close()
    if ret:
        return processes, base_level_actions, termination_times_list, deadlines_list

def copy_dir(src_folder, num_of_samples):
    for src_input in os.listdir(src_folder):
        if len(src_input) < 5 or src_input[-4:] != ".txt":
            src_input += ".txt"
        sample_and_add_to_input(src_folder, src_input, num_of_samples)

def run_sampling(src_folder, num_of_samples):
    copy_dir(src_folder, num_of_samples)

if __name__ == "__main__":
    src_folder = "15-puzzle-multiple-sampling/helper_folder2"
    num_of_samples = 100
    run_sampling(src_folder, num_of_samples)