import os
import numpy as np
from entorno_navegacion import Navegacion
from representacion import FeedbackConstruction
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import time

class SarsaAgent:
    """
    SarsaAgent is an implementation of the SARSA(0) algorithm for reinforcement learning.
    Attributes:
        env (gym.Env): The environment in which the agent operates.
        feedback (object): An object that processes observations from the environment.
        learning_rate (float): The learning rate for updating the weights.
        discount_factor (float): The discount factor for future rewards.
        epsilon (float): The probability of choosing a random action (exploration rate).
        num_actions (int): The number of possible actions in the environment.
        feature_size (int): The size of the feature vector for each state.
        weights (list of np.ndarray): The weights for each action.
    Methods:
        __init__(env, gateway, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
            Initializes the SarsaAgent with the given parameters.
        get_action(state, epsilon=None):
            Returns an action based on the epsilon-greedy policy.
        get_q_values(state):
            Computes the Q-values for all actions given the current state.
        update(state, action, reward, next_state, next_action):
            Updates the weights based on the SARSA update rule.
        train(num_episodes):
            Trains the agent for a specified number of episodes.
        evaluate(num_episodes):
            Evaluates the agent's performance over a specified number of episodes.
    """
    def __init__(self, env, feedback, learning_rate=0.5, discount_factor=0.9, epsilon=0.5):
        # Mejor no toques estas líneas
        self.env = env
        self.feedback = feedback
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = env.action_space.n
        self.feature_size = feedback.iht.size # Si vas a añadir más variables (features) además del tile coding, resérvales espacio aquí.
        ##############################

        # Te damos los pesos inicializados a cero. Pero esto es arbitrario. Lo puedes cambiar si quieres.
        self.weights = [np.zeros(self.feature_size) for _ in range(self.num_actions)]
        
        # Tendrás que usar estrategias para monitorizar el aprendizaje del agente.
        # Añade aquí los atributos que necesites para hacerlo.
        self.episode_rewards = []
        self.episode_lengths = []
        self.sum_of_values = []
        self.grid_value_history = {}
        
        # Define grid positions to track (x, y coordinates)
        self.tracked_positions = [
            (1.0, 1.0),   # bottom-left corner
            (3.5, 1.0),
            (6.5, 1.0),
            (9.0, 1.0),
            (6.0, 5.0),   # center
            (9.0, 9.0),   # top-right corner
            (3.5, 9.0),   # near target area
        ]
        
        # Initialize history for each position
        for pos in self.tracked_positions:
            self.grid_value_history[pos] = []


        ##############################

    def get_action(self, state, epsilon=None):
        """
        Selects an action based on the epsilon-greedy policy.
        Parameters:
        state (object): The current state of the environment.
        epsilon (float, optional): The probability of selecting a random action. 
                                   If None, the default epsilon value is used.
        Returns:
        int: The selected action.
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            return self.env.action_space.sample()  # Random action
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def get_q_values(self, state):
        """
        Computes the Q-values of all actions for a given state.

        Parameters:
        state (object): The current state for which Q-values need to be computed.

        Returns:
        np.ndarray: A numpy array of Q-values for each action in the given state.
        """
        features = self.feedback.process_observation(state)
        q_values = np.array([0,0,0,0])
        # Calcula los valores de cada acción para el estado dado como argumento (aproximación lineal). Añade aquí tu código

        for action in range(self.num_actions):
            q_values[action] = np.sum(self.weights[action][features])
    

        ###################################
        return q_values
    
    def update(self, state, action, reward, next_state, next_action, terminated):
        """
        Update the weights for the given state-action pair using the SARSA(0) algorithm.
        Parameters:
        state (object): The current state.
        action (int): The action taken in the current state.
        reward (float): The reward received after taking the action.
        next_state (object): The state resulting from taking the action.
        next_action (int): The action to be taken in the next state.
        Returns:
        None
        """
        qs_current = self.get_q_values(state)       
        q_current = qs_current[action]
        # td_error
        if terminated:
            td_error = reward - q_current
        else:
            qs_next = self.get_q_values(next_state)
            q_next = qs_next[next_action]            
            td_error = reward + self.discount_factor * q_next - q_current
        # Añade aquí tu código para actualizar los pesos del agente
        features = self.feedback.process_observation(state)
        self.weights[action][features] += self.learning_rate * td_error
        
        #############################################
        
    def train(self, num_episodes, decay_start=0.3, decay_rate=0.7, min_epsilon=0.1, plot=True):
        """
        Train the agent using the SARSA(0) algorithm.
        Parameters:
        num_episodes (int): The number of episodes to train the agent.
        The method runs the training loop for the specified number of episodes.
        In each episode, the agent interacts with the environment, selects actions
        based on the current policy, and updates the policy using the SARSA(0) update rule.
        The total reward for each episode is printed every 100 episodes.
        Returns:
        None
        """
        #Juega con estos tres hiperparámetros

        decay_start = .3 #entre 0 y 1. 
        decay_rate = .995 #control del decrecimiento (exponencial) de epsilon
        min_epsilon = .01 #valor mínimo de epsilon
        # si le bajas el epsilon a 0 el agente depende ya solo de sus decisiones
        

        for episode in range(num_episodes):
            #Set-up del episodio
            state, _ = self.env.reset()
            #Decrecimiento exponencial de epsilon hasta valor mínimo desde comienzo marcado
            if episode >= num_episodes*decay_start:
                self.epsilon *= decay_rate
                self.epsilon = np.max([min_epsilon,self.epsilon])
            #Primera acción
            action = self.get_action(state, self.epsilon)            
            n_steps = 0
            #Generación del episodio
            total_undiscounted_return = 0
            while True:                                        
                next_state, reward, terminated, truncated, _ = self.env.step(action)  
                total_undiscounted_return += reward          
                next_action = self.get_action(next_state, self.epsilon)
                self.update(state, action, reward, next_state, next_action, terminated)    
                state = next_state
                action = next_action                
                n_steps += 1
                if terminated or truncated:
                    break
                
            self.episode_rewards.append(total_undiscounted_return)
            self.episode_lengths.append(n_steps)
            total_sum = 0
            for action in range(agent.num_actions):
                total_sum += np.sum(agent.weights[action])
            self.sum_of_values.append(total_sum)
            
            for pos in self.tracked_positions:
                # Create a synthetic observation for this position
                synthetic_state = np.array([pos[0], pos[1], 0, 0])  # x, y, collision, target_area
                q_values = self.get_q_values(synthetic_state)
                max_q = np.max(q_values)
                self.grid_value_history[pos].append(max_q)
            
            

            #Aquí también puedes cambiar la frecuencia con la que muestras
            #los resultados en la consola, e incluso deshabilitarla.
            episodes_update = 1000
            if episode % episodes_update == 0:                      
                print(f"Episode {episode}, Total undiscounted return: {total_undiscounted_return}, Epsilon: {self.epsilon}") # a futuro cambiar el total_undiscounted_return por media de últimos episodios
                #puedes salvar el estado actual del agente, si te viene bien    
                
        if plot:
            self.plot_training_metrics()

    
    def evaluate(self, num_episodes):
        """
        Evaluate the agent's performance over a specified number of episodes.
        Parameters:
        num_episodes (int): The number of episodes to run the evaluation.
        Returns:
        float: The average reward obtained over the specified number of episodes.
        This method runs the agent in the environment for a given number of episodes
        using a greedy policy (epsilon=0). It collects the total reward for each episode
        and computes the average reward over all episodes. The average reward is printed
        and returned.
        Note:
        - The environment is reset at the beginning of each episode.
        - The agent's action is determined by the `get_action` method with epsilon set to 0.
        """
        total_returns = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_undiscounted_return = 0
            terminated = False
            
            while not terminated:
                action = self.get_action(state, epsilon=0.01)  # Greedy policy
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.env.render()
                state = next_state
                total_undiscounted_return += reward
            
            total_returns.append(total_undiscounted_return)
        
        avg_return = np.mean(total_returns)
        print(f"Average undiscounted return over {num_episodes} episodes: {avg_return}")
        return avg_return

    def plot_training_metrics(self, window_size=100):
        """
        Generate and save plots for all tracked training metrics.
        
        Parameters:
        window_size (int): Window size for computing moving averages
        """
        # Create timestamp and folder structure
        timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        class_name = self.__class__.__name__
        save_dir = os.path.join('runs_graphs', f"{class_name}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Episode Rewards
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_rewards, alpha=0.3, label='Raw Rewards')
        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(self.episode_rewards, 
                                     np.ones(window_size)/window_size, 
                                     mode='valid')
            plt.plot(range(window_size-1, len(self.episode_rewards)), 
                    moving_avg, 
                    label=f'Moving Average (window={window_size})', 
                    linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards Over Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'episode_rewards.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Episode Lengths
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_lengths, alpha=0.3, label='Raw Episode Length')
        if len(self.episode_lengths) >= window_size:
            moving_avg = np.convolve(self.episode_lengths, 
                                     np.ones(window_size)/window_size, 
                                     mode='valid')
            plt.plot(range(window_size-1, len(self.episode_lengths)), 
                    moving_avg, 
                    label=f'Moving Average (window={window_size})', 
                    linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Length Over Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'episode_lengths.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Rewards vs Total Timesteps
        if hasattr(self, 'episode_timesteps') and len(self.episode_timesteps) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(self.episode_timesteps, self.episode_rewards, alpha=0.3, label='Raw Rewards')
            if len(self.episode_rewards) >= window_size:
                moving_avg = np.convolve(self.episode_rewards, 
                                         np.ones(window_size)/window_size, 
                                         mode='valid')
                plt.plot(self.episode_timesteps[window_size-1:], 
                        moving_avg, 
                        label=f'Moving Average (window={window_size})', 
                        linewidth=2)
            plt.xlabel('Total Timesteps')
            plt.ylabel('Total Reward')
            plt.title('Learning Curve: Rewards vs Total Experience')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'rewards_vs_timesteps.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Sum of Values
        if hasattr(self, 'sum_of_values') and len(self.sum_of_values) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(self.sum_of_values)
            plt.xlabel('Episode')
            plt.ylabel('Sum of All Weights')
            plt.title('Sum of All Q-Value Weights Over Training')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'sum_of_values.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 5. Grid Value History
        if hasattr(self, 'grid_value_history') and len(self.grid_value_history) > 0:
            plt.figure(figsize=(14, 7))
            for pos, values in self.grid_value_history.items():
                plt.plot(values, label=f'Position {pos}', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Max Q-Value')
            plt.title('Value Function Evolution at Tracked Grid Positions')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'grid_value_history.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to: {save_dir}")
        return save_dir


if __name__ == "__main__":
    #instanciamos entorno, representación y agente
    #No tocar
    env = Navegacion()
    warehouse_width = 10.0
    warehouse_height = 10.0
    ################
    #diseñar los tiles
    n_tiles_width = 20
    n_tiles_height = 20
    n_tilings = 8
    
    target_area = (2.5, 8, 1.0, 2.0)

    feedback = FeedbackConstruction((warehouse_width, warehouse_height), 
                                 (n_tiles_width, n_tiles_height), 
                                 n_tilings, target_area)
    
    # Crear y entrenar el primer agente (SARSA-10k)
    agent = SarsaAgent(env, feedback, learning_rate=0.1, discount_factor=0.99, epsilon=0.5)
    agent.train(num_episodes=10_000, decay_start=0.3, decay_rate=0.995, min_epsilon=0.01)

    # Guardar agente con 10k episodios
    os.makedirs('agents', exist_ok=True)
    filename_10k = 'agente_sarsa_10k.pkl'
    with open(os.path.join('agents', filename_10k), 'wb') as f:
        pickle.dump(agent, f)

    print("\n--- Evaluación del agente SARSA-10k ---")
    agent.evaluate(num_episodes=1000)

    # Continuar entrenamiento (SARSA-full)
    agent = SarsaAgent(env, feedback, learning_rate=0.1, discount_factor=0.99, epsilon=0.5)
    agent.train(num_episodes=50_000, decay_start=0.3, decay_rate=0.9995, min_epsilon=0.01)

    # Guardar el agente más entrenado
    filename_full = 'agente_sarsa_50k.pkl'
    with open(os.path.join('agents', filename_full), 'wb') as f:
        pickle.dump(agent, f)

    print("\n--- Evaluación del agente SARSA-full ---")
    agent.evaluate(num_episodes=1000)

    
    # agent = SarsaAgent(env, feedback, learning_rate=0.1, discount_factor=0.99, epsilon=0.5)
    
    # # Train the agent
    # agent.train(num_episodes=10000)
    
    # # save the agent object into memory inside `agents/`
    # os.makedirs('agents', exist_ok=True)
    # filename = 'agente_grupo_06_b_1000.pkl'
    # with open(os.path.join('agents', filename), 'wb') as f:
    #     pickle.dump(agent, f)

    # # Evaluate the agent
    # agent.evaluate(num_episodes=1)


