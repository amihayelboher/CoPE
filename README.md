# CoPE

A Formal Metareasoning Model of Concurrent Planning and Execution.

Note that the names of the problem and three of the algorithms were change, but the names were not changed in the code or files.
The problem was called IPAE, and now it's called CoPE.
Max-LET is called here Laziness;
Refined-Max-LET is called here JustDoSomething;
Demand-Execution is called here Executive.

The simplest way to run the project is as follows:
(0. Download the code. No special library installation is needed.)
1. Enter the tests_part2 folder/15-puzzle-multiple-sampling/helper_folder2.
2. Unless you changed anything, the 20-processes instances that were used for the experiments are currently there. Otherwise, put there your CoPE instances.
3. Open a command line in the tests_part2 folder.
4. If you changed anything or put another instances, run the command "python add_tts_and_dls_to_input.py" to sample a 100(-or some another number) deadlines and termination times.
5. Run the command "python ipae_tests_part_2.py".

Some words about the files and directories, just to get a notion:
- tests_part2: Testing file. Make sure to comment out the MDP line in the list of algorithms if running instances with more than 2 processes.
- ipae_heuristics: Contains the algorithms and some more functions.
- add_tts_and_dls_to_input: Sampling deadlines and termination time for CoPE instances in the helper_folder2 folder, and create CoPE instances. The new instances appear in the helper_folder folder.
- results_x: The results for the x-processes instances appear in this directory.
- tile_puzzle_15_a_star: Create data for the (D_i and M_i) histograms.
- create_m_lists: Use the (M_i) histogram data in order to create the M_i distributions.
- mdp: Used for the CoPE MDP implementation. Source (was changed a bit): https://github.com/aimacode/aima-python/blob/master/mdp.py.
