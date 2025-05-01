# === pbt_trainer.py ===
import os
import random
import torch
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

from agents.mlp_agent import SelfLearningAgent
from env.tiny_battlegrounds import TinyBattlegroundsEnv

# ===== Create folders if not exist =====
os.makedirs("saved_models", exist_ok=True)

# === Sample states for plotting
SAMPLE_STATE = torch.tensor([
    0, 10, 5, 20, 9,
    140, 7,
    4, 4, 5, 3, 2, 0,
    20, 5, 120,
    0, 0, 0, 0
], dtype=torch.float32)

SAMPLE_STATE_WEAK = torch.tensor([
    6, 10, 3, 10, 8,
    9, 3,
    2, 1, 3, 0, 0, 0,
    15, 5, 70,
    0, 0, 0, 0
], dtype=torch.float32)

SAMPLE_STATE_WEAK2 = torch.tensor([
    0, 10, 3, 10, 8,
    0, 0,
    2, 1, 3, 0, 0, 0,
    15, 5, 70,
    0, 0, 0, 0
], dtype=torch.float32)

V_HISTORY = []
V_HISTORY_WEAK = []
V_HISTORY_WEAK2 = []

# ===== Core Functions =====
def exploit_explore(agents, genealogy_graph, top_k=5, replace_fraction=0.2):
    agents.sort(key=lambda x: x.mmr, reverse=True)
    top_agents = agents[:top_k]
    num_to_replace = int(len(agents) * replace_fraction)

    for i in range(1, num_to_replace + 1):
        agent_to_replace = agents[-i]
        parent = random.choice(top_agents)

        agent_to_replace.policy.load_state_dict(parent.policy.state_dict())
        agent_to_replace.lr = parent.lr
        agent_to_replace.optimizer = torch.optim.Adam(
            agent_to_replace.policy.parameters(), lr=agent_to_replace.lr
        )
        agent_to_replace.ancestor = parent.ancestor
        agent_to_replace.mutate(max_mmr=parent.mmr)

        genealogy_graph.add_edge(parent.ancestor, agent_to_replace.name)

    print(f"🔁 Mid-gen replace: {num_to_replace} agents refreshed from Top-{top_k}")


def evolve_population(agents, genealogy_graph, generation, inject_every=5, inject_fraction=0.2):
    agents.sort(key=lambda x: x.mmr, reverse=True)
    survivors = agents[:len(agents) // 2]

    new_agents = []
    max_mmr = max(agent.mmr for agent in agents)

    for survivor in survivors:
        cloned = SelfLearningAgent(20, 5, lr=survivor.lr, name=f"{survivor.name}_clone", ancestor=survivor.ancestor)
        cloned.policy.load_state_dict(survivor.policy.state_dict())
        cloned.mutate(max_mmr)
        new_agents.append(survivor)
        new_agents.append(cloned)
        genealogy_graph.add_edge(survivor.ancestor, cloned.name)

    if generation % inject_every == 0:
        num_inject = int(len(new_agents) * inject_fraction)
        for _ in range(num_inject):
            fresh_agent = SelfLearningAgent(20, 5, name=f"Random_{generation}_{random.randint(0,9999)}")
            new_agents[-1] = fresh_agent

    unique_ancestors = set(agent.ancestor for agent in new_agents)
    print(f"🌳 Unique Ancestors after Generation {generation}: {len(unique_ancestors)} → {list(unique_ancestors)}")
    return new_agents


def tournament_generation(agents, genealogy_graph, games_per_agent=5, generation_num=1):
    env = TinyBattlegroundsEnv(agents)
    for agent in agents:
        agent.mmr = 0

    halfway = games_per_agent // 2
    for _ in range(halfway):
        env.play_game()
        for agent in agents:
            agent.learn(agent.mmr)

    exploit_explore(agents, genealogy_graph)

    for _ in range(games_per_agent - halfway):
        env.play_game()
        for agent in agents:
            agent.learn(agent.mmr)

    return evolve_population(agents, genealogy_graph, generation=generation_num)


def main():
    random.seed(42)
    torch.manual_seed(42)

    agents = [SelfLearningAgent(20, 5, name=f"Bot_{i}") for i in range(32)]
    genealogy_graph = nx.DiGraph()
    generations = 100

    try:
        for generation in range(1, generations + 1):
            agents = tournament_generation(agents, genealogy_graph, games_per_agent=5, generation_num=generation)
            avg_mmr = sum(agent.mmr for agent in agents) // len(agents)
            top_agent = max(agents, key=lambda x: x.mmr)

            with torch.no_grad():
                V_HISTORY.append(top_agent.value_net(SAMPLE_STATE).item())
                V_HISTORY_WEAK.append(top_agent.value_net(SAMPLE_STATE_WEAK).item())
                V_HISTORY_WEAK2.append(top_agent.value_net(SAMPLE_STATE_WEAK2).item())

            print(f"\n=== Generation {generation}: Average MMR = {avg_mmr}, Top Agent = {top_agent.name} (MMR: {top_agent.mmr:.0f}) ===")

    except KeyboardInterrupt:
        print("\n⚡ Training interrupted! Saving agents...")

    finally:
        # Plot value net prediction trend
        plt.figure()
        plt.plot(V_HISTORY, label='Strong State', marker='o')
        plt.plot(V_HISTORY_WEAK, label='Weak State', marker='x')
        plt.plot(V_HISTORY_WEAK2, label='Empty State', marker='.')
        plt.xlabel("Generation")
        plt.ylabel("Predicted V(s)")
        plt.title("ValueNet Prediction Over Generations")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Save top agent
        top_agent = max(agents, key=lambda x: x.mmr)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"saved_models/agent_{top_agent.name}_final_{now}.pt"
        top_agent.save(save_path)
        print(f"✅ Saved top agent: {top_agent.name} to {save_path}")


if __name__ == "__main__":
    main()
