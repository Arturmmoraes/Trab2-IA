import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent
from fish_algo import FishSchoolSearch
from test_trained_agent import test_agent
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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


# --- Função de fitness ---
def game_fitness_function(individual: np.ndarray) -> float:
    game_config = GameConfig(num_players=1, fps=60)
    agent = NeuralNetworkAgent(weights_path=None)
    agent.weights = vector_to_weights(individual)

    total_score = 0
    n_games = 5 # Joga 5 vezes (consegui rodar em menos de 12 horas com 5 jogos, mas é possível diminuir para 3 caso demore muito)
    for _ in range(n_games):
        game = SurvivalGame(config=game_config, render=False)
        while not game.all_players_dead():
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
        total_score += game.players[0].score

    return total_score / n_games

# --- Treinamento e Teste da Rede Neural com metaheurística Cardume de Peixes ---
def train_and_test_fss_agent():
    n_fish = 100
    n_iterations = 1000
    weight_bounds = (-0.5, 0.5)
    step_ind_init = 0.1
    step_vol_init = 1
    step_ind_final = 0.001
    step_vol_final = 0.01

    fss = FishSchoolSearch(
        fitness_func=game_fitness_function,
        n_fish=n_fish,
        n_iterations=n_iterations,
        weight_bounds=weight_bounds,
        step_ind_init=step_ind_init,
        step_vol_init=step_vol_init,
        step_ind_final=step_ind_final,
        step_vol_final=step_vol_final,
        input_dim=INPUT_DIM
    )

    best_weights, best_fitness = fss.evolve()
    print(f"\nTraining finished! Best fitness: {best_fitness:.2f}")
    np.save("best_weights.npy", vector_to_weights(best_weights))
    total_scores = test_agent(best_weights, num_tests=30, render=True)
    return total_scores

def resultados_boxplot(resultados):
    sns.boxplot(data=resultados)
    plt.xlabel("Método/Agente")
    plt.ylabel("Score")
    plt.title("Distribuição dos Resultados por Método/Agente")
    plt.show()

if __name__ == "__main__":
    total_scores = train_and_test_fss_agent()

    rule_based_result = [12.69, 16.65, 6.97, 2.79, 15.94, 10.22, 21.90, 4.35, 6.22, 9.95, 19.94, 20.56, 15.74, 17.68, 7.16, 15.68, 2.37, 
    15.43, 15.13, 22.50, 25.82, 15.85, 17.02, 16.74, 14.69, 11.73, 13.80, 15.13, 12.35, 16.19]

    neural_agent_result = [38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30, 39.76, 48.17, 44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 
    67.17, 53.54, 33.59, 49.24, 52.65, 16.35, 44.05, 56.59, 63.23, 43.96, 43.82, 19.19, 28.36, 18.65]

    human_result = [27.34, 17.63, 39.33, 17.44, 1.16, 24.04, 29.21,   18.92, 25.71, 20.05, 31.88, 15.39, 22.50, 19.27, 26.33, 23.67, 16.82, 
    28.45, 12.59, 33.01, 21.74, 14.23, 27.90, 24.80, 11.35, 30.12, 17.08, 22.96, 9.41, 35.22]

    df = pd.DataFrame({
    "FSS": total_scores,
    "Rule-based": rule_based_result,
    "Neural GA": neural_agent_result,
    "Human": human_result
    })
    df.loc["Média"] = df.mean()
    df.loc["Desvio Padrão"] = df.std()

    print(df.to_string(float_format="%.2f"))

    resultados = {
    "FSS": total_scores,
    "Rule-based": rule_based_result,
    "Neural GA": neural_agent_result,
    "Human": human_result
    }

    nomes = list(resultados.keys())
    n = len(nomes)

    matriz = [["" for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                matriz[i][j] = nomes[i]
            elif i < j:
                _, p = ttest_ind(resultados[nomes[i]], resultados[nomes[j]], equal_var=False)
                if p <= 0.05:
                    valor = f"*{p:.5f}*"
                else:
                    valor = f"{p:.5f}"
                matriz[i][j] = valor
            elif i > j:
                t, p_w = stats.wilcoxon(
                    resultados[nomes[i]],
                    resultados[nomes[j]]
                )
                #p_w = p_w.round(3)
                if p_w <= 0.05:
                    valor = f"*{p_w:.5f}*"
                else:
                    valor = f"{p_w:.5f}"
                matriz[i][j] = valor

    df_pvalues = pd.DataFrame(matriz)

    print(df_pvalues.to_string())

    df_treinamento = pd.read_csv("fss_history.csv")

    x = df_treinamento.iloc[:,0]
    y = df_treinamento.iloc[:,1]

    plt.figure(figsize=(10,6))
    plt.plot(x, y, label="Best Fitness", linewidth=1.8)

    plt.xlabel("Iterações")
    plt.ylabel("Best Fitness")
    plt.title("Evolução do agente FSS")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    resultados_boxplot(resultados)