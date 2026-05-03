import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random
import copy

# ==================== DATA LOADING AND PREPROCESSING ====================
def load_and_preprocess_data():
    """Load digits dataset and preprocess"""
    X, y = load_digits(return_X_y=True)
    y = y.reshape(-1, 1)

    enc = OneHotEncoder(sparse_output=False)
    y_onehot = enc.fit_transform(y)

    X_train, X_temp, y_train_raw, y_temp_raw, y_train_onehot, y_temp_onehot = train_test_split(
        X, y, y_onehot, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val_raw, y_test_raw, y_val_onehot, y_test_onehot = train_test_split(
        X_temp, y_temp_raw, y_temp_onehot, test_size=0.5, random_state=42
    )

    # Normalization
    X_train = X_train / 16.0
    X_val = X_val / 16.0
    X_test = X_test / 16.0

    return X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw, y_train_onehot, y_val_onehot, y_test_onehot

def extract_features(X):
    """
    Feature extraction function - for digits dataset, we use raw pixels
    For image datasets, you can use pretrained models here (VGG, ResNet, etc.)
    """
    return X

# ==================== ACTIVATION FUNCTIONS ====================
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x**2

def get_activation(name):
    activations = {
        'sigmoid': (sigmoid, sigmoid_derivative),
        'relu': (relu, relu_derivative),
        'tanh': (tanh, tanh_derivative)
    }
    return activations[name]

# ==================== NEURAL NETWORK FROM SCRATCH ====================
class NeuralNetworkScratch:
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01, optimizer='sgd'):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.activation, self.activation_derivative = get_activation(activation)
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            if activation == 'relu':
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        # For Adam optimizer
        if optimizer == 'adam':
            self.m_weights = [np.zeros_like(w) for w in self.weights]
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
            self.t = 0

        # For RMSProp optimizer
        if optimizer == 'rmsprop':
            self.cache_weights = [np.zeros_like(w) for w in self.weights]
            self.cache_biases = [np.zeros_like(b) for b in self.biases]

        # For Adagrad optimizer
        if optimizer == 'adagrad':
            self.cache_weights = [np.zeros_like(w) for w in self.weights]
            self.cache_biases = [np.zeros_like(b) for b in self.biases]

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if i == len(self.weights) - 1:  # Output layer
                a = sigmoid(z)
            else:
                a = self.activation(z)

            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[0]
        deltas = []

        # Output layer error
        delta = self.activations[-1] - y
        deltas.append(delta)

        # Hidden layers errors
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(deltas[0], self.weights[i+1].T) * self.activation_derivative(self.activations[i+1])
            deltas.insert(0, delta)

        # Calculate gradients
        weight_gradients = []
        bias_gradients = []

        for i in range(len(self.weights)):
            dw = np.dot(self.activations[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m

            # Gradient clipping
            dw = np.clip(dw, -5, 5)
            db = np.clip(db, -5, 5)

            weight_gradients.append(dw)
            bias_gradients.append(db)

        return weight_gradients, bias_gradients

    def update_weights(self, weight_gradients, bias_gradients):
        if self.optimizer == 'sgd':
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * weight_gradients[i]
                self.biases[i] -= self.learning_rate * bias_gradients[i]

        elif self.optimizer == 'adam':
            self.t += 1
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8

            for i in range(len(self.weights)):
                # Update biased first moment estimate
                self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * weight_gradients[i]
                self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * bias_gradients[i]

                # Update biased second moment estimate
                self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (weight_gradients[i]**2)
                self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (bias_gradients[i]**2)

                # Compute bias-corrected moment estimates
                m_w_hat = self.m_weights[i] / (1 - beta1**self.t)
                m_b_hat = self.m_biases[i] / (1 - beta1**self.t)
                v_w_hat = self.v_weights[i] / (1 - beta2**self.t)
                v_b_hat = self.v_biases[i] / (1 - beta2**self.t)

                # Update weights
                self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        elif self.optimizer == 'rmsprop':
            decay_rate = 0.9
            epsilon = 1e-8

            for i in range(len(self.weights)):
                self.cache_weights[i] = decay_rate * self.cache_weights[i] + (1 - decay_rate) * (weight_gradients[i]**2)
                self.cache_biases[i] = decay_rate * self.cache_biases[i] + (1 - decay_rate) * (bias_gradients[i]**2)

                self.weights[i] -= self.learning_rate * weight_gradients[i] / (np.sqrt(self.cache_weights[i]) + epsilon)
                self.biases[i] -= self.learning_rate * bias_gradients[i] / (np.sqrt(self.cache_biases[i]) + epsilon)

        elif self.optimizer == 'adagrad':
            epsilon = 1e-8

            for i in range(len(self.weights)):
                self.cache_weights[i] += weight_gradients[i]**2
                self.cache_biases[i] += bias_gradients[i]**2

                self.weights[i] -= self.learning_rate * weight_gradients[i] / (np.sqrt(self.cache_weights[i]) + epsilon)
                self.biases[i] -= self.learning_rate * bias_gradients[i] / (np.sqrt(self.cache_biases[i]) + epsilon)

    def train(self, X, y, X_val, y_val, epochs, batch_size, verbose=False):
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Forward pass
                self.forward(X_batch)

                # Backward pass
                weight_gradients, bias_gradients = self.backward(X_batch, y_batch)

                # Update weights
                self.update_weights(weight_gradients, bias_gradients)

            # Calculate losses and accuracies
            train_pred = self.forward(X)
            train_loss = -np.mean(np.sum(y * np.log(train_pred + 1e-8), axis=1))
            train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y, axis=1))

            val_pred = self.forward(X_val)
            val_loss = -np.mean(np.sum(y_val * np.log(val_pred + 1e-8), axis=1))
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        return train_losses, val_losses, train_accs, val_accs

    def predict(self, X):
        """Predict function that returns class predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=1)

def predict(model, X_test):
    """Standalone predict function for a single test sample"""
    if len(X_test.shape) == 1:
        X_test = X_test.reshape(1, -1)

    prediction = model.predict(X_test)
    print(f"Predicted class: {prediction[0]}")
    return prediction[0]

def plot_results(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

# ==================== BUILT-IN MODEL (PyTorch Alternative) ====================
class NeuralNetworkBuiltin:
    """Simple numpy implementation simulating built-in behavior"""
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01, optimizer='adam'):
        self.model = NeuralNetworkScratch(layer_sizes, activation, learning_rate, optimizer)

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        return self.model.train(X, y, X_val, y_val, epochs, batch_size, verbose=False)

    def predict(self, X):
        return self.model.predict(X)

# ==================== GENETIC ALGORITHM ====================
class GeneticAlgorithm:
    def __init__(self, population_size=10, generations=5, mutation_rate=0.2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_solution = None
        self.best_fitness = 0
        self.history = []

        # Hyperparameter search space
        self.hidden_layers_options = [1, 2, 3, 4, 5]
        self.neurons_options = [32, 64, 128, 256, 512]
        self.activation_options = ['relu', 'tanh', 'sigmoid']
        self.lr_options = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        self.batch_size_options = [16, 32, 64, 128]
        self.optimizer_options = ['sgd', 'adam', 'rmsprop', 'adagrad']
        self.epochs_options = list(range(3, 21))

    def create_random_solution(self):
        """Create a random hyperparameter configuration"""
        n_hidden = random.choice(self.hidden_layers_options)
        return {
            'hidden_layers': n_hidden,
            'neurons': [random.choice(self.neurons_options) for _ in range(n_hidden)],
            'activation': random.choice(self.activation_options),
            'learning_rate': random.choice(self.lr_options),
            'batch_size': random.choice(self.batch_size_options),
            'optimizer': random.choice(self.optimizer_options),
            'epochs': random.choice(self.epochs_options)
        }

    def fitness_function(self, solution, X_train, y_train, X_val, y_val):
        """Evaluate a hyperparameter configuration"""
        try:
            input_size = X_train.shape[1]
            output_size = y_train.shape[1]
            layer_sizes = [input_size] + solution['neurons'] + [output_size]

            model = NeuralNetworkScratch(
                layer_sizes=layer_sizes,
                activation=solution['activation'],
                learning_rate=solution['learning_rate'],
                optimizer=solution['optimizer']
            )

            _, _, _, val_accs = model.train(
                X_train, y_train, X_val, y_val,
                epochs=solution['epochs'],
                batch_size=solution['batch_size'],
                verbose=False
            )

            fitness = max(val_accs)
            return fitness
        except Exception as e:
            print(f"Error in fitness function: {e}")
            return 0.0

    def crossover(self, parent1, parent2):
        """Create offspring by combining two parents"""
        child = {}

        # Randomly choose from parents
        for key in parent1.keys():
            if key == 'neurons':
                # Handle neurons list specially
                n_layers = random.choice([parent1['hidden_layers'], parent2['hidden_layers']])
                parent_neurons = parent1['neurons'] if len(parent1['neurons']) >= n_layers else parent2['neurons']
                child['neurons'] = parent_neurons[:n_layers]
                child['hidden_layers'] = n_layers
            elif key != 'hidden_layers':
                child[key] = random.choice([parent1[key], parent2[key]])

        return child

    def mutate(self, solution):
        """Randomly mutate a solution"""
        mutated = copy.deepcopy(solution)

        if random.random() < self.mutation_rate:
            key = random.choice(list(mutated.keys()))

            if key == 'hidden_layers':
                mutated['hidden_layers'] = random.choice(self.hidden_layers_options)
                mutated['neurons'] = [random.choice(self.neurons_options) for _ in range(mutated['hidden_layers'])]
            elif key == 'neurons':
                idx = random.randint(0, len(mutated['neurons']) - 1)
                mutated['neurons'][idx] = random.choice(self.neurons_options)
            elif key == 'activation':
                mutated['activation'] = random.choice(self.activation_options)
            elif key == 'learning_rate':
                mutated['learning_rate'] = random.choice(self.lr_options)
            elif key == 'batch_size':
                mutated['batch_size'] = random.choice(self.batch_size_options)
            elif key == 'optimizer':
                mutated['optimizer'] = random.choice(self.optimizer_options)
            elif key == 'epochs':
                mutated['epochs'] = random.choice(self.epochs_options)

        return mutated

    def optimize(self, X_train, y_train, X_val, y_val):
        """Run genetic algorithm optimization"""
        print(f"\n{'='*60}")
        print("GENETIC ALGORITHM OPTIMIZATION")
        print(f"{'='*60}")

        # Initialize population
        population = [self.create_random_solution() for _ in range(self.population_size)]

        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")

            # Evaluate fitness for all solutions
            fitness_scores = []
            for i, solution in enumerate(population):
                fitness = self.fitness_function(solution, X_train, y_train, X_val, y_val)
                fitness_scores.append(fitness)
                print(f"  Solution {i+1}: Fitness = {fitness:.4f}")

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = copy.deepcopy(solution)

            self.history.append({
                'generation': generation,
                'best_fitness': max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores)
            })

            print(f"  Best fitness: {max(fitness_scores):.4f}, Avg fitness: {np.mean(fitness_scores):.4f}")

            # Selection (tournament selection)
            new_population = []

            # Elitism - keep best solution
            best_idx = np.argmax(fitness_scores)
            new_population.append(copy.deepcopy(population[best_idx]))

            # Create rest of new population
            while len(new_population) < self.population_size:
                # Tournament selection
                idx1, idx2 = random.sample(range(len(population)), 2)
                parent1 = population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]

                idx1, idx2 = random.sample(range(len(population)), 2)
                parent2 = population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                child = self.mutate(child)

                new_population.append(child)

            population = new_population

        print(f"\n{'='*60}")
        print(f"BEST SOLUTION FOUND: Fitness = {self.best_fitness:.4f}")
        print(f"{'='*60}")
        print(f"Configuration: {self.best_solution}")

        return self.best_solution, self.best_fitness

# ==================== PARTICLE SWARM OPTIMIZATION ====================
class ParticleSwarmOptimization:
    def __init__(self, n_particles=10, iterations=5, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.best_solution = None
        self.best_fitness = 0
        self.history = []

        # Hyperparameter search space (encoded as continuous values)
        self.bounds = {
            'hidden_layers': (1, 5),
            'neurons': (32, 512),
            'activation': (0, 2),  # 0=relu, 1=tanh, 2=sigmoid
            'learning_rate': (-5, -1),  # log scale
            'batch_size': (0, 3),  # 0=16, 1=32, 2=64, 3=128
            'optimizer': (0, 3),  # 0=sgd, 1=adam, 2=rmsprop, 3=adagrad
            'epochs': (3, 20)
        }

    def decode_solution(self, position):
        """Convert continuous position to discrete hyperparameters"""
        activation_map = {0: 'relu', 1: 'tanh', 2: 'sigmoid'}
        batch_size_map = {0: 16, 1: 32, 2: 64, 3: 128}
        optimizer_map = {0: 'sgd', 1: 'adam', 2: 'rmsprop', 3: 'adagrad'}

        n_hidden = int(np.clip(position[0], 1, 5))

        return {
            'hidden_layers': n_hidden,
            'neurons': [int(np.clip(position[1], 32, 512)) for _ in range(n_hidden)],
            'activation': activation_map[int(np.clip(position[2], 0, 2))],
            'learning_rate': 10 ** np.clip(position[3], -5, -1),
            'batch_size': batch_size_map[int(np.clip(position[4], 0, 3))],
            'optimizer': optimizer_map[int(np.clip(position[5], 0, 3))],
            'epochs': int(np.clip(position[6], 3, 20))
        }

    def create_random_position(self):
        """Create random particle position"""
        return np.array([
            random.uniform(1, 5),  # hidden_layers
            random.uniform(32, 512),  # neurons
            random.uniform(0, 2),  # activation
            random.uniform(-5, -1),  # learning_rate (log)
            random.uniform(0, 3),  # batch_size
            random.uniform(0, 3),  # optimizer
            random.uniform(3, 20)  # epochs
        ])

    def fitness_function(self, position, X_train, y_train, X_val, y_val):
        """Evaluate fitness of a particle position"""
        try:
            solution = self.decode_solution(position)

            input_size = X_train.shape[1]
            output_size = y_train.shape[1]
            layer_sizes = [input_size] + solution['neurons'] + [output_size]

            model = NeuralNetworkScratch(
                layer_sizes=layer_sizes,
                activation=solution['activation'],
                learning_rate=solution['learning_rate'],
                optimizer=solution['optimizer']
            )

            _, _, _, val_accs = model.train(
                X_train, y_train, X_val, y_val,
                epochs=solution['epochs'],
                batch_size=solution['batch_size'],
                verbose=False
            )

            fitness = max(val_accs)
            return fitness
        except Exception as e:
            print(f"Error in fitness function: {e}")
            return 0.0

    def optimize(self, X_train, y_train, X_val, y_val):
        """Run PSO optimization"""
        print(f"\n{'='*60}")
        print("PARTICLE SWARM OPTIMIZATION")
        print(f"{'='*60}")

        # Initialize particles
        particles = [self.create_random_position() for _ in range(self.n_particles)]
        velocities = [np.zeros(7) for _ in range(self.n_particles)]

        # Personal best
        personal_best_positions = copy.deepcopy(particles)
        personal_best_fitness = [0.0] * self.n_particles

        # Global best
        global_best_position = None
        global_best_fitness = 0.0

        for iteration in range(self.iterations):
            print(f"\nIteration {iteration + 1}/{self.iterations}")

            for i in range(self.n_particles):
                # Evaluate fitness
                fitness = self.fitness_function(particles[i], X_train, y_train, X_val, y_val)
                print(f"  Particle {i+1}: Fitness = {fitness:.4f}")

                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = copy.deepcopy(particles[i])

                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = copy.deepcopy(particles[i])

            self.history.append({
                'iteration': iteration,
                'best_fitness': global_best_fitness,
                'avg_fitness': np.mean(personal_best_fitness)
            })

            print(f"  Best fitness: {global_best_fitness:.4f}, Avg fitness: {np.mean(personal_best_fitness):.4f}")

            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = random.random(), random.random()

                cognitive = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social = self.c2 * r2 * (global_best_position - particles[i])

                velocities[i] = self.w * velocities[i] + cognitive + social
                particles[i] = particles[i] + velocities[i]

                # Clip to bounds
                particles[i][0] = np.clip(particles[i][0], 1, 5)
                particles[i][1] = np.clip(particles[i][1], 32, 512)
                particles[i][2] = np.clip(particles[i][2], 0, 2)
                particles[i][3] = np.clip(particles[i][3], -5, -1)
                particles[i][4] = np.clip(particles[i][4], 0, 3)
                particles[i][5] = np.clip(particles[i][5], 0, 3)
                particles[i][6] = np.clip(particles[i][6], 3, 20)

        self.best_solution = self.decode_solution(global_best_position)
        self.best_fitness = global_best_fitness

        print(f"\n{'='*60}")
        print(f"BEST SOLUTION FOUND: Fitness = {self.best_fitness:.4f}")
        print(f"{'='*60}")
        print(f"Configuration: {self.best_solution}")

        return self.best_solution, self.best_fitness

# ==================== MAIN EXECUTION ====================
def main():
    print("="*60)
    print("ML COURSE PROJECT - NEURAL NETWORK WITH HYPERPARAMETER OPTIMIZATION")
    print("="*60)

    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw, y_train_onehot, y_val_onehot, y_test_onehot = load_and_preprocess_data()

    # Extract features
    X_train_features = extract_features(X_train)
    X_val_features = extract_features(X_val)
    X_test_features = extract_features(X_test)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # Run Genetic Algorithm
    print("\n2. Running Genetic Algorithm...")
    ga = GeneticAlgorithm(population_size=8, generations=3, mutation_rate=0.2)
    ga_best_solution, ga_best_fitness = ga.optimize(X_train_features, y_train_onehot, X_val_features, y_val_onehot)

    # Run PSO
    print("\n3. Running Particle Swarm Optimization...")
    pso = ParticleSwarmOptimization(n_particles=8, iterations=3)
    pso_best_solution, pso_best_fitness = pso.optimize(X_train_features, y_train_onehot, X_val_features, y_val_onehot)

    # Train final models with best hyperparameters
    print("\n4. Training final models with optimized hyperparameters...")

    # GA Model (From Scratch)
    print("\n4a. Training model with GA hyperparameters (From Scratch)...")
    input_size = X_train_features.shape[1]
    output_size = y_train_onehot.shape[1]
    ga_layer_sizes = [input_size] + ga_best_solution['neurons'] + [output_size]

    ga_model = NeuralNetworkScratch(
        layer_sizes=ga_layer_sizes,
        activation=ga_best_solution['activation'],
        learning_rate=ga_best_solution['learning_rate'],
        optimizer=ga_best_solution['optimizer']
    )

    ga_train_losses, ga_val_losses, ga_train_accs, ga_val_accs = ga_model.train(
        X_train_features, y_train_onehot, X_val_features, y_val_onehot,
        epochs=ga_best_solution['epochs'],
        batch_size=ga_best_solution['batch_size'],
        verbose=True
    )

    # PSO Model (From Scratch)
    print("\n4b. Training model with PSO hyperparameters (From Scratch)...")
    pso_layer_sizes = [input_size] + pso_best_solution['neurons'] + [output_size]

    pso_model = NeuralNetworkScratch(
        layer_sizes=pso_layer_sizes,
        activation=pso_best_solution['activation'],
        learning_rate=pso_best_solution['learning_rate'],
        optimizer=pso_best_solution['optimizer']
    )

    pso_train_losses, pso_val_losses, pso_train_accs, pso_val_accs = pso_model.train(
        X_train_features, y_train_onehot, X_val_features, y_val_onehot,
        epochs=pso_best_solution['epochs'],
        batch_size=pso_best_solution['batch_size'],
        verbose=True
    )

    # Built-in Model (using best hyperparameters from GA)
    print("\n4c. Training built-in model...")
    builtin_model = NeuralNetworkBuiltin(
        layer_sizes=ga_layer_sizes,
        activation=ga_best_solution['activation'],
        learning_rate=ga_best_solution['learning_rate'],
        optimizer=ga_best_solution['optimizer']
    )

    builtin_train_losses, builtin_val_losses, builtin_train_accs, builtin_val_accs = builtin_model.train(
        X_train_features, y_train_onehot, X_val_features, y_val_onehot,
        epochs=ga_best_solution['epochs'],
        batch_size=ga_best_solution['batch_size']
    )

    # Plot results
    print("\n5. Plotting results...")
    plot_results(ga_train_losses, ga_val_losses, ga_train_accs, ga_val_accs)

    # Test prediction function
    print("\n6. Testing predict function...")
    test_sample = X_test_features[0]
    print(f"True class: {y_test_raw[0][0]}")
    predict(ga_model, test_sample)

    # Evaluate on test set
    print("\n7. Evaluating models on test set...")

    # GA Model
    ga_test_pred = ga_model.predict(X_test_features)
    ga_test_acc = np.mean(ga_test_pred == y_test_raw.ravel())
    print(f"\nGA Model Test Accuracy: {ga_test_acc*100:.2f}%")
    print("\nGA Model Classification Report:")
    print(classification_report(y_test_raw.ravel(), ga_test_pred))

    # PSO Model
    pso_test_pred = pso_model.predict(X_test_features)
    pso_test_acc = np.mean(pso_test_pred == y_test_raw.ravel())
    print(f"\nPSO Model Test Accuracy: {pso_test_acc*100:.2f}%")
    print("\nPSO Model Classification Report:")
    print(classification_report(y_test_raw.ravel(), pso_test_pred))

    # Built-in Model
    builtin_test_pred = builtin_model.predict(X_test_features)
    builtin_test_acc = np.mean(builtin_test_pred == y_test_raw.ravel())
    print(f"\nBuilt-in Model Test Accuracy: {builtin_test_acc*100:.2f}%")
    print("\nBuilt-in Model Classification Report:")
    print(classification_report(y_test_raw.ravel(), builtin_test_pred))

    # Plot confusion matrices
    print("\n8. Creating confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # GA Confusion Matrix
    cm_ga = confusion_matrix(y_test_raw.ravel(), ga_test_pred)
    sns.heatmap(cm_ga, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'GA Model\nAccuracy: {ga_test_acc*100:.2f}%')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # PSO Confusion Matrix
    cm_pso = confusion_matrix(y_test_raw.ravel(), pso_test_pred)
    sns.heatmap(cm_pso, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'PSO Model\nAccuracy: {pso_test_acc*100:.2f}%')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    # Built-in Confusion Matrix
    cm_builtin = confusion_matrix(y_test_raw.ravel(), builtin_test_pred)
    sns.heatmap(cm_builtin, annot=True, fmt='d', cmap='Oranges', ax=axes[2])
    axes[2].set_title(f'Built-in Model\nAccuracy: {builtin_test_acc*100:.2f}%')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

   

    print("\n--- Genetic Algorithm Results ---")
    print(f"Best Hyperparameters: {ga_best_solution}")
    print(f"Validation Accuracy: {ga_best_fitness*100:.2f}%")
    print(f"Test Accuracy: {ga_test_acc*100:.2f}%")

    print("\n--- PSO Results ---")
    print(f"Best Hyperparameters: {pso_best_solution}")
    print(f"Validation Accuracy: {pso_best_fitness*100:.2f}%")
    print(f"Test Accuracy: {pso_test_acc*100:.2f}%")

    print("\n--- Built-in Model Results ---")
    print(f"Test Accuracy: {builtin_test_acc*100:.2f}%")

    print("\n--- Best Model ---")
    best_acc = max(ga_test_acc, pso_test_acc, builtin_test_acc)
    if best_acc == ga_test_acc:
        print("Genetic Algorithm achieved the best test accuracy!")
    elif best_acc == pso_test_acc:
        print("PSO achieved the best test accuracy!")
    else:
        print("Built-in model achieved the best test accuracy!")

    print(f"Best Test Accuracy: {best_acc*100:.2f}%")


    print("\nGenerated files:")
    print("  - training_results.png (Loss and accuracy plots)")
    print("  - confusion_matrices.png (Confusion matrices for all models)")

if __name__ == "__main__":
    main()