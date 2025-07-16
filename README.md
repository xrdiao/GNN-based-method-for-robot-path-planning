# GNN-based-method-for-robot-path-planning

Implementation code for our paper "[Graph neural network based method for robot path planning](https://www.sciencedirect.com/science/article/pii/S2667379724000056)" in Biomimetic Intelligence and Robotics 2024. This repository contains our GNN-based method for robot path planning code for training and testing the planning policy in its PyBullet simulator.

## Introduction
We propose a learning-based path planning method that reduces the number of collision checks. We develop an efficient neural network model based on graph neural networks. The model outputs weights for each neighbor based on the obstacle, searched path, and random geometric graph, which are used to guide the planner in avoiding obstacles. We evaluate the efficiency of the proposed path planning method through simulated random worlds and real-world experiments. The results demonstrate that the proposed method significantly reduces the number of collision checks and improves the path planning speed in high-dimensional environments.

## Citation

```bibtex
@article{DIAO2024100147,
title = {Graph neural network based method for robot path planning},
journal = {Biomimetic Intelligence and Robotics},
volume = {4},
number = {1},
pages = {100147},
year = {2024},
issn = {2667-3797},
doi = {https://doi.org/10.1016/j.birob.2024.100147},
url = {https://www.sciencedirect.com/science/article/pii/S2667379724000056},
author = {Xingrong Diao and Wenzheng Chi and Jiankun Wang},
}
```

