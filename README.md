# Hierarchical Clustering: O(1)-Approximation for Well-Clustered Graphs
This repository contains code to accompany the paper "Hierarchical Clustering: O(1)-Approximation for Well-Clustered Graphs", 
published in NeurIPS 2021. It provides an implementation of the proposed algorithm PruneMerge and several other algorithms
for hierarchical clustering. Below you can find instructions for running the code.

## Requirements
The algorithm was implemented in Python 3.8.
To install the requirements of this project, run the command:
```setup
pip install -r requirements.txt
```

## Training
No training is required since the algorithm is unsupervised.


## Evaluation
To evaluate the performance of the algorithm, run the command:
```eval
python eval.py {experiment_type}
```

where {experiment_type} is the type of experiment to be run. This must be one of the following options:
['complete_graph', 'SBM_standard', 'SBM_planted_clique', 'HSBM', 'real_datasets'] 

## Results
The results will be displayed in a newly created output file "Results_{datetime}.txt", where {datetime} is the current
date and time. For every experiment and every hierarchical tree construction, 
the exact and approximated Dasgupta's cost is printed.
