# === simulate_game.py ===
import torch
from env.tiny_battlegrounds import TinyBattlegroundsEnv
from agents.mlp_agent import SelfLearningAgent  
from env.tiny_battlegrounds import MINION_POOL
from agents.transformer_agent import TransformerAgent

AGENT_TYPE = "transformer"  # or "transformer"

def make_agent(i):
    if AGENT_TYPE == "transformer":
        return TransformerAgent(name=f"Transformer_{i}")
    elif AGENT_TYPE == "mlp":
        return SelfLearningAgent(input_size=20, action_size=5, name=f"MLP_{i}")
    else:
        raise ValueError(f"Unsupported AGENT_TYPE: {AGENT_TYPE}")



def simulate_single_game(agent):
    dummy_agents = [SelfLearningAgent(20, 5, name=f"Dummy_{i}") for i in range(7)]
    for dummy in dummy_agents:
        dummy.policy.eval()

    all_agents = dummy_agents + [agent]

    env = TinyBattlegroundsEnv(all_agents)

    print("\n=== Starting Single Game Simulation ===\n")
    env.play_game(verbose=True, focus_agent_name=agent.name)  # 🔥 Focus only on your agent

    print("\n=== Game Over ===")
    for a in env.agents:
        if a.name == agent.name:
            print(f"\n{a.name} Final Health: {a.health}")
            print("Final Board:")
            for m in a.board:
                print(f"  {m.name} ({m.attack}/{m.health}) {'*' * m.tier}")



if __name__ == "__main__":
    # ✨ Load your trained agent
    agent = make_agent(0)
    agent.load("saved_models/BEST_Transformer_6_transformer_20250502_031217.pt")  # Example
    simulate_single_game(agent)
