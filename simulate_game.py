# === simulate_game.py ===
import torch
from tiny_battlegrounds import TinyBattlegroundsEnv
from tiny_pbt import SelfLearningAgent  # Import your agent class
from tiny_battlegrounds import MINION_POOL
def simulate_single_game(agent):
    dummy_agents = [SelfLearningAgent(9, 5, name=f"Dummy_{i}") for i in range(7)]
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
    agent = SelfLearningAgent(9, 5, name="MyTestAgent")
    agent.load("saved_models/agent_Bot_27_clone_clone_clone_final_20250429_195355.pt")  # Example
    print(MINION_POOL)
    simulate_single_game(agent)
