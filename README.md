# NeuroEvolve: Neural Networks with Genetic and Swarm Optimization

This project implements a custom Neural Network from scratch and uses evolutionary algorithms—Genetic Algorithm (GA) and Particle Swarm Optimization (PSO)—to find the optimal hyperparameters.

## Overview
Instead of manually tuning hyperparameters, this project explores the search space using two powerful optimization techniques:
- **Genetic Algorithm (GA)**: Evolves a population of hyperparameter configurations using selection, crossover, and mutation.
- **Particle Swarm Optimization (PSO)**: Simulates a swarm of particles moving through the hyperparameter space to find the global best configuration.

The optimized hyperparameters include:
- Number of hidden layers
- Neurons per layer
- Activation function (ReLU, Tanh, Sigmoid)
- Learning Rate
- Batch Size
- Optimizer (SGD, Adam, RMSprop, Adagrad)
- Epochs

## Results & Visualizations
The script trains the models on the `digits` dataset and outputs:
- Best hyperparameter configurations found by GA and PSO.
- Comparison of Test Accuracies between the GA-optimized model, PSO-optimized model, and a simulated Built-in Model.
- `training_results.png`: Plots of training/validation loss and accuracy.
- `confusion_matrices.png`: Confusion matrices comparing the performance of the models.
