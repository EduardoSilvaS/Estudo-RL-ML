import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.std_dev = 1

    def pull(self):
        """ Retorna uma recompensa baseada na média verdadeira + ruído aleatório. """
        return np.random.randn() + self.true_mean

    def drift(self):
        """ Adiciona uma pequena mudança aleatória à recompensa média verdadeira. """
        # Sorteia um pequeno valor e o adiciona à média.
        self.true_mean += np.random.randn() * 0.05

class Agent:
    def __init__(self, num_bandits, epsilon):
        self.epsilon = epsilon
        self.q_estimates = np.zeros(num_bandits)
        self.n_selections = np.zeros(num_bandits)

    def choose_action(self):
        """ Escolhe uma ação usando a estratégia epsilon-greedy. """
        # Sorteia um número aleatório entre 0 e 1
        rand_num = np.random.random()

        if rand_num < self.epsilon:
            # EXPLORAR: escolhe um bandit aleatoriamente
            return np.random.randint(len(self.q_estimates))
        else:
            # EXPLOTAR: escolhe o bandit com a maior estimativa de recompensa
            return np.argmax(self.q_estimates)

    def update(self, action, reward):
        """ Atualiza a estimativa de Q-value para a ação tomada. """
        # Incrementa o contador para a ação escolhida
        self.n_selections[action] += 1
        
        # Fórmula da média incremental: Q_novo = Q_antigo + (1/N) * (Recompensa - Q_antigo)
        # É uma forma eficiente de recalcular a média sem guardar todas as recompensas.
        self.q_estimates[action] += (1.0 / self.n_selections[action]) * (reward - self.q_estimates[action])


def run_simulation(num_bandits, num_steps, epsilon):
    # Define as recompensas médias iniciais para cada bandit.
    # Vamos criar valores bem definidos para que haja um "melhor" claro.
    initial_means = np.random.uniform(low=0, high=5, size=num_bandits)
    bandits = [Bandit(mean) for mean in initial_means]
    
    agent = Agent(num_bandits=num_bandits, epsilon=epsilon)
    
    # Listas para guardar dados para os gráficos
    rewards_history = []
    actions_history = []
    
    print(f"Recompensas médias verdadeiras iniciais: {[round(b.true_mean, 2) for b in bandits]}")

    for step in range(num_steps):
        # 1. Agente escolhe uma ação
        action = agent.choose_action()
        
        # 2. Ambiente retorna uma recompensa para a ação
        reward = bandits[action].pull()
        
        # 3. Agente atualiza suas estimativas com base na recompensa
        agent.update(action, reward)
        
        # Guarda os resultados
        rewards_history.append(reward)
        actions_history.append(action)
        
        # 4. Simula o drift a cada 100 passos para testar a re-exploração
        if step % 100 == 0 and step > 0:
            for bandit in bandits:
                bandit.drift()
    
    print(f"Estimativas finais do agente: {[round(q, 2) for q in agent.q_estimates]}")
    print(f"Recompensas médias verdadeiras finais: {[round(b.true_mean, 2) for b in bandits]}")
    
    return rewards_history, actions_history, agent.n_selections

# -----------------------------------------------------------------------------
# PASSO 4: ANÁLISE E VISUALIZAÇÃO
# -----------------------------------------------------------------------------
# Documentação:
# Esta função usa o `matplotlib` para criar os diagnósticos visuais.
# - Gráfico 1: Recompensa por Passo. Mostra a recompensa obtida em cada passo.
#   Inclui uma "média móvel" para visualizar a tendência de aprendizado do agente.
# - Gráfico 2: Taxa de Seleção por Botão. Um gráfico de barras que mostra
#   quantas vezes cada botão foi escolhido. Idealmente, o botão com a maior
#   recompensa real deveria ser o mais escolhido.
def plot_results(rewards, selections, num_bandits):
    # Define o tamanho da janela para a média móvel
    window_size = 100
    
    # Calcula a média móvel das recompensas
    moving_average = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    # Cria a figura e os eixos para os gráficos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))
    fig.suptitle('Resultados da Simulação Caça-Luzes (ε-greedy)', fontsize=16)

    # Gráfico 1: Recompensa por Passo
    ax1.plot(moving_average)
    ax1.set_title('Recompensa Média Móvel (Janela de 100 passos)')
    ax1.set_xlabel('Passos da Simulação')
    ax1.set_ylabel('Recompensa Média')
    ax1.grid(True)

    # Gráfico 2: Taxa de Seleção por Botão
    ax2.bar(range(num_bandits), selections)
    ax2.set_title('Número de Vezes que Cada Botão Foi Escolhido')
    ax2.set_xlabel('Botão (Bandit)')
    ax2.set_ylabel('Número de Seleções')
    ax2.set_xticks(range(num_bandits)) # Garante que todos os números dos botões apareçam no eixo x

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# -----------------------------------------------------------------------------
# PONTO DE ENTRADA PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Parâmetros que você pode ajustar ---
    NUM_BANDITS = 5      # Quantos botões teremos (entre 4 e 6)
    NUM_STEPS = 20000     # Total de vezes que vamos pressionar um botão
    EPSILON = 0.1        # Taxa de exploração (10% de chance de explorar)
    
    # Executa a simulação
    rewards_history, actions_history, selections = run_simulation(
        num_bandits=NUM_BANDITS,
        num_steps=NUM_STEPS,
        epsilon=EPSILON
    )
    
    # Plota os resultados
    plot_results(rewards_history, selections, NUM_BANDITS)
