# tune_hyperparams.py
from entorno_navegacion import Navegacion
from representacion import FeedbackConstruction
from agente import SarsaAgent
import numpy as np
import pickle

def make_agent(n_tiles_width=20, n_tiles_height=20, n_tilings=8,
               alpha=0.1, gamma=0.99, epsilon0=0.5):
    env = Navegacion()
    feedback = FeedbackConstruction((10.0, 10.0),
                                    (n_tiles_width, n_tiles_height),
                                    n_tilings,
                                    target_area=(2.5, 8.0, 1.0, 2.0))
    agent = SarsaAgent(env, feedback,
                       learning_rate=alpha,
                       discount_factor=gamma,
                       epsilon=epsilon0)
    return agent

if __name__ == "__main__":
    list_decay_start = [0.5, 0.6, 0.7]
    list_decay_rate  = [0.98, 0.99, 0.995]
    list_min_epsilon = [0.01, 0.05, 0.1]

    train_episodes = 5000      # rápido para comparar; sube luego
    eval_episodes  = 20        # evaluación corta para ranking inicial

    best_score = -np.inf
    best_cfg   = None

    for ds in list_decay_start:
        for dr in list_decay_rate:
            for me in list_min_epsilon:
                agent = make_agent()
                agent.train(num_episodes=train_episodes,
                            decay_start=ds, decay_rate=dr, min_epsilon=me)
                # evaluación 100% greedy (ajusta agente.evaluate a epsilon=0.0)
                avg = agent.evaluate(num_episodes=eval_episodes)
                print(f"[ds={ds}, dr={dr}, me={me}] -> avg_return={avg:.2f}")

                if avg > best_score:
                    best_score = avg
                    best_cfg = (ds, dr, me)
                    with open(f"best_agent_temp_DAVID.pkl", "wb") as f:
                        pickle.dump(agent, f)

    print(f"\nMEJOR: ds={best_cfg[0]} dr={best_cfg[1]} me={best_cfg[2]}  "
          f"con avg_return={best_score:.2f}")

    # entrenar “a conciencia” con la mejor combinación y guardar definitivo
    best_agent = make_agent()
    best_agent.train(num_episodes=10000,
                     decay_start=best_cfg[0], decay_rate=best_cfg[1], min_epsilon=best_cfg[2])
    with open(f"agente_best_ds{best_cfg[0]}_dr{best_cfg[1]}_me{best_cfg[2]}.pkl", "wb") as f:
        pickle.dump(best_agent, f)
