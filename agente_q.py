import numpy as np
from entorno_navegacion import Navegacion
from representacion import FeedbackConstruction
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

class QLearningAgent:
    """
    QLearningAgent implements the Q-learning algorithm with linear function approximation using tile coding.
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
            Initializes the Q-learning with the given parameters.
        get_action(state, epsilon=None):
            Returns an action based on the epsilon-greedy policy.
        get_q_values(state):
            Computes the Q-values for all actions given the current state.
        update(state, action, reward, next_state, next_action):
            Updates the weights based on the Q-learning update rule.
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
        self.feature_size = feedback.iht.size  # Si vas a añadir más variables (features) además del tile coding, resérvales espacio aquí.
        ##############################

        # Te damos los pesos inicializados a cero. Pero esto es arbitrario. Lo puedes cambiar si quieres.
        self.weights = [np.zeros(self.feature_size, dtype=np.float64) for _ in range(self.num_actions)]
        
        # Tendrás que usar estrategias para monitorizar el aprendizaje del agente.
        # Añade aquí los atributos que necesites para hacerlo.
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timesteps = []

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
            return self.env.action_space.sample()  # Acción aleatoria
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
        features = np.asarray(self.feedback.process_observation(state), dtype=np.intp)
        features = np.mod(features, self.feature_size)
        q_values = np.zeros(self.num_actions, dtype=np.float64)
        for a in range(self.num_actions):
            q_values[a] = self.weights[a][features].sum(dtype=np.float64)

        ###################################
        return q_values
    
    def update(self, state, action, reward, next_state, next_action, terminated):
        """
        Update the weights for the given state-action pair using the Q-learning algorithm.
        Parameters:
        state (object): The current state.
        action (int): The action taken in the current state.
        reward (float): The reward received after taking the action.
        next_state (object): The state resulting from taking the action.
        next_action (int): The action to be taken in the next state.
        Returns:
        None
        """
        # Q(s,a)
        q_current = self.get_q_values(state)[action]

        # Objetivo Q-learning: target = r           si terminal
        #                       r + gamma * max_a' Q(s', a')  en caso contrario
        if terminated:
            target = reward
        else:
            q_next_max = np.max(self.get_q_values(next_state))
            target = reward + self.discount_factor * q_next_max

        td_error = target - q_current

        # Actualización de pesos en los features activos de (s,a)
        features = np.asarray(self.feedback.process_observation(state), dtype=np.intp)
        features = np.mod(features, self.feature_size)

        # step por feature (evita explosiones con varios tilings) + clipping de seguridad
        eta = self.learning_rate / max(1, len(features))
        td_error = np.clip(td_error, -1e3, 1e3)

        delta = eta * td_error
        self.weights[action][features] = np.clip(
            self.weights[action][features] + delta,
            -1e6, 1e6
        )
        
        #############################################
        
    def train(self, num_episodes, decay_start=0.3, decay_rate=0.7, min_epsilon=0.1):
        """
        Train the agent using the Q-learning algorithm.
        Parameters:
        num_episodes (int): The number of episodes to train the agent.
        The method runs the training loop for the specified number of episodes.
        In each episode, the agent interacts with the environment, selects actions
        based on the current policy, and updates the policy using the Q-learning update rule.
        The total reward for each episode is printed every 100 episodes.
        Returns:
        None
        """
        # Juega con estos tres hiperparámetros
        decay_start = .5  # entre 0 y 1.
        decay_rate = .99  # control del decrecimiento (exponencial) de epsilon
        min_epsilon = .1  # valor mínimo de epsilon
        # si le bajas el epsilon a 0 el agente depende ya solo de sus decisiones
        list_decay_start = [0.2, 0.3, 0.4]
        list_decay_rate = [0.9, 0.95, 0.99]
        list_min_epsilon = [0.01, 0.05, 0.1]

        for episode in range(num_episodes):
            # Set-up del episodio
            state, _ = self.env.reset()

            # Decrecimiento exponencial de epsilon hasta valor mínimo desde comienzo marcado
            if episode >= num_episodes * decay_start:
                self.epsilon *= decay_rate
                self.epsilon = np.max([min_epsilon, self.epsilon])

            # Primera acción (epsilon-greedy)
            action = self.get_action(state, self.epsilon)
            n_steps = 0

            # Generación del episodio
            total_undiscounted_return = 0
            while True:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_undiscounted_return += reward

                # (Q-learning) La actualización usa max_a' Q(s', a'), por lo que next_action no es necesario.
                # Mantenemos la firma y pasamos un placeholder (None).
                self.update(state, action, reward, next_state, next_action=None, terminated=terminated)

                # Siguiente iteración: elegimos acción para actuar (comportamiento sigue siendo epsilon-greedy)
                state = next_state
                action = self.get_action(state, self.epsilon)

                n_steps += 1
                if terminated or truncated:
                    break

            # Aquí también puedes cambiar la frecuencia con la que muestras
            # los resultados en la consola, e incluso deshabilitarla.
            episodes_update = 1000
            if episode % episodes_update == 0:
                print(f"Episode {episode}, Total undiscounted return: {total_undiscounted_return}, Epsilon: {self.epsilon}")  # a futuro cambiar el total_undiscounted_return por media de últimos episodios
                # puedes salvar el estado actual del agente, si te viene bien    

    
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
                action = self.get_action(state, epsilon=0.01)  # Política casi-greedy (puedes poner 0.0 si quieres greedy puro)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.env.render()
                state = next_state
                total_undiscounted_return += reward
            
            total_returns.append(total_undiscounted_return)
        
        avg_return = np.mean(total_returns)
        print(f"Average undiscounted return over {num_episodes} episodes: {avg_return}")
        return avg_return


if __name__ == "__main__":
    # instanciamos entorno, representación y agente
    # No tocar
    env = Navegacion()
    warehouse_width = 10.0
    warehouse_height = 10.0
    ################
    # diseñar los tiles
    n_tiles_width = 20
    n_tiles_height = 20
    n_tilings = 8
    
    target_area = (2.5, 8, 1.0, 2.0)

    feedback = FeedbackConstruction((warehouse_width, warehouse_height), 
                                 (n_tiles_width, n_tiles_height), 
                                 n_tilings, target_area)
    
    agent = QLearningAgent(env, feedback, learning_rate=0.1, discount_factor=0.99, epsilon=0.5)
    
    # Train the agent
    agent.train(num_episodes=10000)
    
    # save the agent object into memory    
    with open('agente_q_grupo_06_b.pkl', 'wb') as f:
        pickle.dump(agent, f)

    # Evaluate the agent
    agent.evaluate(num_episodes=1)
