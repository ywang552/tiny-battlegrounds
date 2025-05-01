import torch
from transformer_agent import TransformerAgent
from tiny_battlegrounds import TinyBattlegroundsEnv

def run_test_game(num_agents=8, verbose=True, focus_agent_name="TAgent_0"):
    print("🚀 Launching test game...")

    agents = [TransformerAgent(name=f"TAgent_{i}") for i in range(num_agents)]
    env = TinyBattlegroundsEnv(agents)

    rewards = env.play_game(verbose=verbose, focus_agent_name=focus_agent_name)

    print("\n=== Final Rewards ===")
    for name, reward in rewards.items():
        print(f"{name}: {reward:.1f}")

    print("\n=== Final MMRs ===")
    for agent in agents:
        print(f"{agent.name}: {agent.mmr:.1f}")

    print("\n=== Opponent Memory Sample (First Agent) ===")
    if isinstance(agents[0], TransformerAgent):
        for k, v in agents[0].opponent_memory.items():
            print(f"Opponent {k}: {v}")

    return agents, env


if __name__ == "__main__":
    agents, env = run_test_game()
