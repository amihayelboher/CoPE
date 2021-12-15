from add_tts_and_dls_to_input import run_sampling
from ipae_tests_part_2 import run_tests


##########################################################################################################
# Preferred not to use this module but divide the testing to run_sampling first and just then run_tests! #
# By running this module, the files are not available until the run is finished, while by running        #
# run_sampling the files are created and become available (initially empty) and then the results are     #
# available while running run_tests.                                                                     #
##########################################################################################################

if __name__ == "__main__":
    folders = [
        "15-puzzle-multiple-sampling/helper_folder2",
    ]
    num_of_samples = 1
    for src_folder in folders:
        run_sampling(src_folder, num_of_samples)
        run_tests()