"""
This script evaluates the performance of the Algorithm PruneMerge against other classical and state-of-the-art
algorithms for hierarchical clustering.
"""

import experiments
import datetime
import time
import sys
import argparse

# The type of trees considered
tree_types = ["average_linkage", "single_linkage", "complete_linkage", "Prune_Merge"]
# The type of experiments considered
experiment_types = ['complete_graph', 'SBM_standard', 'SBM_planted_clique', 'HSBM', 'real_datasets']


def main():
    sys.setrecursionlimit(10000)
    out = set_output_file()
    args = parse_args()

    if args.experiment == 'complete_graph':
        experiments.run_experiment_complete_graph(tree_types, out)
    elif args.experiment == 'SBM_standard':
        experiments.run_experiment_SBM(tree_types, out)
    elif args.experiment == 'SBM_planted_clique':
        experiments.run_experiment_SBM_planted_clique(tree_types, out)
    elif args.experiment == 'HSBM':
        experiments.run_experiment_HSBM(tree_types, out)
    elif args.experiment == 'real_datasets':
        experiments.run_experiment_real_data(tree_types, out)
    else:
        raise Exception(f'Unknown experiment')

    out.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('experiment', type=str, choices=experiment_types, help="which experiment to perform")
    return parser.parse_args()


def set_output_file():
    filename = str("Results_" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H_%M_%S') + ".txt")
    out = open(filename, "w")
    return out


if __name__ == "__main__":
    main()
