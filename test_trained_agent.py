import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent
import os
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#É chamada no arquivo train_fss_agent.py
def test_agent(weights: np.ndarray, num_tests: int = 30, render: bool = False):
    print(f"\n--- Testando Agente Treinado por {num_tests} vezes ---")
    
    # Configurações do jogo para o teste
    game_config = GameConfig(render_grid=True)

    total_scores = []

    for i in range(num_tests):
        game = SurvivalGame(config=game_config, render=render)
        agent = NeuralNetworkAgent("best_weights.npy")
        
        while not game.all_players_dead():
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            
            game.update([action])
            if render:
                game.render_frame()
        
        final_score = game.players[0].score
        total_scores.append(final_score)
        print(f"Teste {i+1}/{num_tests}: Score Final = {final_score:.2f}")

    avg_score = np.mean(total_scores)
    std_score = np.std(total_scores)
    print(total_scores)
    print(f"\nResultados Finais após {num_tests} testes:")
    print(f"Score Médio: {avg_score:.2f}")
    print(f"Desvio Padrão do Score: {std_score:.2f}")
    return total_scores

#Função parecida com a test_agent, mas sem precisar do weights como parâmetro
#É chamada diretamente no arquivo test_trained_agent.py
def run_agent(num_tests: int = 30, render: bool = True):
    config = GameConfig(render_grid=True)
    total_scores = []
    for i in range(num_tests):
        game = SurvivalGame(config=config, render=True)
        agent = NeuralNetworkAgent("best_weights.npy")

        while not game.all_players_dead():
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            if render:
                game.render_frame()
    
        final_score = game.players[0].score
        total_scores.append(final_score)
        print(f"Teste {i+1}/{num_tests}: Score Final = {final_score:.2f}")

    avg_score = np.mean(total_scores)
    std_score = np.std(total_scores)
    print(total_scores)
    print(f"\nResultados Finais após {num_tests} testes:")
    print(f"Score Médio: {avg_score:.2f}")
    print(f"Desvio Padrão do Score: {std_score:.2f}")
    return total_scores

def resultados_boxplot(resultados):
    sns.boxplot(data=resultados)
    plt.xlabel("Método/Agente")
    plt.ylabel("Score")
    plt.title("Distribuição dos Resultados por Método/Agente")
    plt.show()

if __name__ == "__main__":

    total_scores = run_agent(num_tests=30, render=True)

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

