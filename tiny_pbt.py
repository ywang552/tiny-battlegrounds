# === tiny_pbt_v2_genealogy.py ===
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx  # 🔥 Genealogy Tracking
from datetime import datetime

from tiny_battlegrounds import TinyBattlegroundsEnv

# ===== Create folders if not exist =====
os.makedirs("saved_models", exist_ok=True)

# ===== Models =====
class TinyNNPolicy(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.net(x)

    def predict_action(self, state):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1)
        return action.item(), probs


class TinyValueNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # outputs a scalar V(s)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape: [batch] or scalar



class SelfLearningAgent:
    def __init__(self, input_size, action_size, lr=1e-3, name="Agent", ancestor=None):
        self.name = name
        self.policy = TinyNNPolicy(input_size, action_size)
        self.value_net = TinyValueNet(input_size)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()), lr=lr
        )
        self.lr = lr
        self.memory = []  # stores (state, action, reward, next_state)
        self.mmr = 0.0
        self.cumulative_reward = 0.0
        self.last_cumulative_reward = 0.0
        self.ancestor = ancestor if ancestor is not None else name

    def act(self, state):
        action, probs = self.policy.predict_action(state)
        log_prob = torch.log(probs[action])
        value = self.value_net(state)

        # Store log_prob, value, and current state
        self.memory.append({
            "state": state.detach(),
            "action": action,
            "log_prob": log_prob,
            "value": value,
            # note: next_state and reward added later
        })
        self.last_cumulative_reward = self.cumulative_reward
        return action

    def observe(self, next_state, reward):
        # Add next_state and reward to the latest memory entry
        if self.memory:
            self.memory[-1]["next_state"] = next_state.detach()
            self.memory[-1]["reward"] = reward

    def learn(self, final_mmr):
        if not self.memory:
            return

        gamma = 0.95
        λ = 0.1

        policy_losses = []
        value_losses = []

        for entry in self.memory:
            r = entry["reward"] + λ * final_mmr
            s = entry["state"]
            s_ = entry["next_state"]
            a = entry["action"]
            log_prob = entry["log_prob"]
            value = entry["value"]

            with torch.no_grad():
                next_value = self.value_net(s_)

            # Advantage
            advantage = (r + gamma * next_value) - value

            # Losses
            policy_losses.append(-log_prob * advantage.detach())
            value_losses.append(advantage.pow(2))

        loss = torch.stack(policy_losses).sum() + 0.5 * torch.stack(value_losses).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.clear()

    def mutate(self, max_mmr):
        base_strength = 0.05
        mutation_strength = base_strength * (1 + (max_mmr - self.mmr) / (max_mmr + 1e-8))

        with torch.no_grad():
            for param in self.policy.parameters():
                param.add_(mutation_strength * torch.randn_like(param))
            for param in self.value_net.parameters():
                param.add_(mutation_strength * torch.randn_like(param))

        if random.random() < 0.5:
            self.lr *= random.uniform(0.8, 1.2)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def save(self, filepath):
        torch.save({
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict()
        }, filepath)

    def load(self, filepath):
        data = torch.load(filepath)
        self.policy.load_state_dict(data["policy"])
        self.value_net.load_state_dict(data["value_net"])


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
        agent_to_replace.optimizer = optim.Adam(agent_to_replace.policy.parameters(), lr=agent_to_replace.lr)
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
        cloned = SelfLearningAgent(9, 5, lr=survivor.lr, name=f"{survivor.name}_clone", ancestor=survivor.ancestor)
        cloned.policy.load_state_dict(survivor.policy.state_dict())
        cloned.mutate(max_mmr)
        new_agents.append(survivor)
        new_agents.append(cloned)

        genealogy_graph.add_edge(survivor.ancestor, cloned.name)

    # ✨ Diversity Rescue
    if generation % inject_every == 0:
        num_inject = int(len(new_agents) * inject_fraction)
        for _ in range(num_inject):
            fresh_agent = SelfLearningAgent(9, 5, name=f"Random_{generation}_{random.randint(0,9999)}")
            new_agents[-1] = fresh_agent

    # ✨ Genealogy Report
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


# ===== Main Runner =====
def main():
    random.seed(42)
    torch.manual_seed(42)

    agents = [SelfLearningAgent(9, 5, name=f"Bot_{i}") for i in range(32)]
    genealogy_graph = nx.DiGraph()

    generations = 20

    try:
        for generation in range(1, generations + 1):
            agents = tournament_generation(agents, genealogy_graph, games_per_agent=5, generation_num=generation)
            avg_mmr = sum(agent.mmr for agent in agents) // len(agents)
            top_agent = max(agents, key=lambda x: x.mmr)

            print(f"\n=== Generation {generation}: Average MMR = {avg_mmr}, Top Agent = {top_agent.name} (MMR: {top_agent.mmr:.0f}) ===")

    except KeyboardInterrupt:
        print("\n⚡ Training interrupted! Saving agents...")

    finally:
        # ✨ Plot genealogy tree
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(genealogy_graph, k=0.5, iterations=100)
        nx.draw(genealogy_graph, pos, with_labels=True, node_size=500, font_size=8, arrows=True)
        plt.title("Genealogy Tree After Training")
        plt.show()

        # ✨ Save top agent
        top_agent = max(agents, key=lambda x: x.mmr)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")  # example format: 20240501_021530
        save_path = f"saved_models/agent_{top_agent.name}_final_{now}.pt"
        top_agent.save(save_path)
        print(f"✅ Saved top agent: {top_agent.name} to {save_path}")



if __name__ == "__main__":
    main()
