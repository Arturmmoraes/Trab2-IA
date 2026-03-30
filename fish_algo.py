import numpy as np
import pandas as pd
import os


# --- Arquitetura da rede ---
INPUT_SIZE = 27
HIDDEN1 = 16
HIDDEN2 = 16
OUTPUT_SIZE = 3
INPUT_DIM = HIDDEN1*INPUT_SIZE + HIDDEN1 + HIDDEN2*HIDDEN1 + HIDDEN2 + OUTPUT_SIZE*HIDDEN2 + OUTPUT_SIZE


# --- Funções de utilidade ---
def vector_to_weights(vector: np.ndarray) -> dict:
    idx = 0
    W1_size = HIDDEN1 * INPUT_SIZE
    b1_size = HIDDEN1
    W2_size = HIDDEN2 * HIDDEN1
    b2_size = HIDDEN2
    W3_size = OUTPUT_SIZE * HIDDEN2
    b3_size = OUTPUT_SIZE

    W1 = vector[idx:idx+W1_size].reshape(HIDDEN1, INPUT_SIZE)
    idx += W1_size
    b1 = vector[idx:idx+b1_size]
    idx += b1_size
    W2 = vector[idx:idx+W2_size].reshape(HIDDEN2, HIDDEN1)
    idx += W2_size
    b2 = vector[idx:idx+b2_size]
    idx += b2_size
    W3 = vector[idx:idx+W3_size].reshape(OUTPUT_SIZE, HIDDEN2)
    idx += W3_size
    b3 = vector[idx:idx+b3_size]

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

class FishSchoolSearch:
    def __init__(self, fitness_func, n_fish, n_iterations, weight_bounds, 
                 step_ind_init, step_ind_final, step_vol_init, step_vol_final, 
                 input_dim, Wscale=5000, seed=None):
        self.fitness_func = fitness_func
        self.n_fish = n_fish
        self.n_iterations = n_iterations
        self.weight_bounds = weight_bounds
        self.step_ind_init = step_ind_init
        self.step_ind_final = step_ind_final
        self.step_vol_init = step_vol_init
        self.step_vol_final = step_vol_final
        self.input_dim = input_dim
        self.Wscale = Wscale
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        self.positions = self.rng.uniform(self.weight_bounds[0], self.weight_bounds[1], (self.n_fish, self.input_dim))
        self.weights = np.ones(self.n_fish) * (self.Wscale / 2)
        self.fitness = np.array([self.fitness_func(pos) for pos in self.positions])
        self.best_fitness = np.max(self.fitness)
        self.best_position = self.positions[np.argmax(self.fitness)].copy()
        self.total_weight_prev = np.sum(self.weights)
        self.step_ind = self.step_ind_init
        self.step_vol = self.step_vol_init

    def individual_movement(self):
        old_positions = self.positions.copy()
        new_positions = []
        new_fitness = []

        for i in range(self.n_fish):
            direction = self.rng.uniform(-1, 1, self.input_dim)  #direção aleatória
            step = self.step_ind * self.rng.uniform(0, 1) * direction 
            candidate = old_positions[i] + step #salva a possível nova posição
            candidate = np.clip(candidate, self.weight_bounds[0], self.weight_bounds[1])
            f_new = self.fitness_func(candidate)

            if f_new > self.fitness[i]: #caso melhore
                new_positions.append(candidate)
                new_fitness.append(f_new)
            else:
                new_positions.append(old_positions[i])
                new_fitness.append(self.fitness[i])

        new_positions = np.array(new_positions)
        new_fitness = np.array(new_fitness)
        delta_f = new_fitness - self.fitness
        self.positions = new_positions
        self.fitness = new_fitness
        return delta_f, old_positions

    def feeding(self, delta_f):
        max_delta = np.max(np.abs(delta_f))
        if max_delta > 0:
            self.weights += (delta_f / max_delta) #mudança de peso proporcional ao ganho de comida
            self.weights = np.clip(self.weights, 1, self.Wscale)

    def collective_instinctive_movement(self, delta_f, old_positions):
        sum_delta = np.sum(np.abs(delta_f))
        if sum_delta == 0:
            return
        displacement = self.positions - old_positions
        direction = np.sum((displacement.T * delta_f).T, axis=0) / sum_delta
        self.positions += direction #movimento coletivo na direção do ganho de comida
        self.positions = np.clip(self.positions, self.weight_bounds[0], self.weight_bounds[1])

    def collective_volitive_movement(self):
        barycenter = np.sum(self.positions.T * self.weights, axis=1) / np.sum(self.weights)
        total_weight = np.sum(self.weights)
        rand_factor = self.rng.uniform(0, 1)

        if total_weight > self.total_weight_prev:
            self.positions -= rand_factor * self.step_vol * (self.positions - barycenter) #movimento em direção ao baricentro
        else:
            self.positions += rand_factor * self.step_vol * (self.positions - barycenter) #movimento em direção oposta ao baricentro

        self.positions = np.clip(self.positions, self.weight_bounds[0], self.weight_bounds[1])
        self.total_weight_prev = total_weight

    def breeding(self):
        strong_fish = np.where(self.weights >= self.Wscale * 0.9)[0]
        if len(strong_fish) < 2:
            return
        i = self.rng.choice(strong_fish) #peixe forte aleatório

        scores = []
        for j in strong_fish:
            if j == i:
                continue
            dist = np.linalg.norm(self.positions[i] - self.positions[j])
            if dist == 0:
                dist = 0.000000001  # para evitar divisão por zero
            score = self.weights[j] / dist
            scores.append((score, j))

        j = max(scores, key=lambda x: x[0])[1] #peixe forte com maior taxa de peso sobre distância

        child_pos = (self.positions[i] + self.positions[j]) / 2
        child_w = (self.weights[i] + self.weights[j]) / 2
        child_fit = self.fitness_func(child_pos)

        weakest = np.argmin(self.weights) #remove o peixe mais fraco
        self.positions[weakest] = child_pos
        self.weights[weakest] = child_w
        self.fitness[weakest] = child_fit

    def evolve(self):
        self.initialize()
        for i in range(self.n_iterations):
            delta_f, old_positions = self.individual_movement()
            self.feeding(delta_f)
            self.collective_instinctive_movement(delta_f, old_positions)
            self.collective_volitive_movement()
            self.breeding()

            # decaimento linear dos passos
            self.step_ind = self.step_ind_init - (self.step_ind_init - self.step_ind_final) * (i+1)/self.n_iterations
            self.step_vol = self.step_vol_init - (self.step_vol_init - self.step_vol_final) * (i+1)/self.n_iterations

            max_idx = np.argmax(self.fitness)
            if self.fitness[max_idx] > self.best_fitness:
                self.best_fitness = self.fitness[max_idx]
                self.best_position = self.positions[max_idx].copy()
                np.save("best_weights.npy", vector_to_weights(self.best_position)) #salva os melhores pesos a cada iteração, caso fitness melhore

            print(f"Iteration {i+1}/{self.n_iterations} | Best fitness: {self.best_fitness:.4f}")
            df = pd.DataFrame([[i+1, self.best_fitness]], columns=["iteration", "best_fitness"])
            df.to_csv("fss_history.csv", mode="a", header=not os.path.exists("fss_history.csv"), index=False) #csv para o grafico

        return self.best_position, self.best_fitness


