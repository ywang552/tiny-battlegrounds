import torch
from models.tiny_nn_policy import SelfLearningAgent
from tiny_battlegrounds import TinyBattlegroundsEnv  # your environment

def load_trained_agent(filepath, input_size=9, action_size=5):
    saved = torch.load(filepath)
    agent = SelfLearningAgent(input_size=input_size, action_size=action_size, name=saved["agent_name"])
    agent.policy.load_state_dict(saved["policy_state_dict"])
    agent.mmr = saved["mmr"]
    return agent

def evaluate_agents(agents, num_games=100):
    env = TinyBattlegroundsEnv(agents)
    placement_counter = {agent.name: 0 for agent in agents}

    for _ in range(num_games):
        env.setup()
        rewards = env.play_game()

        ranked = sorted(rewards.items(), key=lambda x: x[1], reverse=True)

        for place, (agent_name, _) in enumerate(ranked, 1):
            placement_counter[agent_name] += place

    print("\n=== Evaluation Results ===")
    for agent_name, total_place in placement_counter.items():
        avg_place = total_place / num_games
        print(f"{agent_name}: Avg Placement = {avg_place:.2f}")

    return placement_counter

if __name__ == "__main__":
    # --- Load the strong agent ---
    trained_agent = load_trained_agent("saved_models/agent_Bot_5_batch35.pt")

    # --- Create 7 new random agents ---
    random_agents = [SelfLearningAgent(input_size=9, action_size=5, name=f"Random_{i}") for i in range(7)]

    # --- Mix the agents ---
    agents_for_test = random_agents + [trained_agent]

    # --- Evaluate ---
    evaluate_agents(agents_for_test, num_games=100)
