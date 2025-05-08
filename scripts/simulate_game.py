# === simulate_game.py ===
import torch
from env.tiny_battlegrounds import TinyBattlegroundsEnv
from agents.mlp_agent import SelfLearningAgent  
from env.tiny_battlegrounds import MINION_POOL
from agents.transformer_agent import TransformerAgent

AGENT_TYPE = "transformer"  # or "transformer"

def make_agent(i, agent_type = AGENT_TYPE):
    if agent_type == "transformer":
        return TransformerAgent(name=f"Transformer_{i}")
    elif agent_type == "mlp":
        return SelfLearningAgent(input_size=19, action_size=5, name=f"MLP_{i}")
    else:
        raise ValueError(f"Unsupported agent_type: {agent_type}")



def simulate_single_game(agent):
    ## TODO
    # dummy_agents = [SelfLearningAgent(19, 16, name=f"Dummy_{i}") for i in range(7)]
    # for dummy in dummy_agents:
    #     dummy.policy.eval()

    agents = []
    for i in range(7):
        agents.append(make_agent(i, agent_type="transformer"))

    all_agents = agents + [agent]
    agent.name = "target"
    env = TinyBattlegroundsEnv(all_agents)

    print("\n=== Starting Single Game Simulation ===\n")
    rewards = env.play_game(verbose=True, focus_agent_name=agent.name)  # 🔥 Focus only on your agent

    print("\n=== Game Over ===")
    for a in env.agents:
        if a.name == agent.name:
            print(f"\n{a.name} Final Health: {a.health}")
            print("Final Board:")
            for m in a.board:
                print(f"  {m.name} ({m.attack}/{m.health}) {'*' * m.tier}")

    agent.learn(rewards[agent.name])  # ✅ Log learning summary



if __name__ == "__main__":
    # ✨ Load your trained agent
    agent = make_agent(0)
    agent.load("saved_models/BEST_Transformer_2_transformer_20250505_020248.pt")  # Example
    simulate_single_game(agent)
    print("\n=== Game Over ===")
    