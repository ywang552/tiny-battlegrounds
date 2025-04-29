# tiny_pbt.py
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import matplotlib.pyplot as plt

# ===== Create folders if not exist =====
os.makedirs("saved_models", exist_ok=True)

# ===== Models =====
class TinyNNPolicy(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, x):
        return self.net(x)

    def predict_action(self, state):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1)
        return action.item(), probs

class SelfLearningAgent:
    def __init__(self, input_size, action_size, lr=1e-3, name="Agent"):
        self.name = name
        self.policy = TinyNNPolicy(input_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.lr = lr
        self.memory = []
        self.cumulative_reward = 0.0

    def act(self, state):
        action, probs = self.policy.predict_action(state)
        log_prob = torch.log(probs[action])
        self.memory.append(log_prob)
        return action

    def learn(self, final_reward):
        if not self.memory:
            return

        rewards = torch.tensor([final_reward for _ in self.memory])
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        loss = torch.stack([-log_prob * reward for log_prob, reward in zip(self.memory, rewards)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []
        self.cumulative_reward = 0.0

    def mutate(self):
        with torch.no_grad():
            for param in self.policy.parameters():
                param.add_(0.02 * torch.randn_like(param))
        # Optionally mutate learning rate
        if random.random() < 0.3:
            self.lr *= random.uniform(0.8, 1.2)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath))

# ===== Environment (placeholder) =====
class TinyBattlegroundsEnv:
    def __init__(self, agents):
        self.agents = agents

    def play_game(self):
        # Fake simple game result for now
        rewards = {agent.name: random.gauss(0, 1) for agent in self.agents}
        for agent in self.agents:
            agent.cumulative_reward += rewards[agent.name]
        return rewards

# ===== PBT Training =====
def tournament_generation(agents, games_per_agent=5, generation_num=1):
    env = TinyBattlegroundsEnv(agents)
    
    # Play games
    for _ in range(games_per_agent):
        env.play_game()

    # Assign MMR
    for agent in agents:
        agent.mmr = agent.cumulative_reward

    # Sort and evolve
    agents.sort(key=lambda x: x.mmr, reverse=True)
    survivors = agents[:len(agents)//2]

    new_agents = []
    for survivor in survivors:
        # Save model
        survivor.save(f"saved_models/{survivor.name}_generation{generation_num}.pt")

        # Clone + mutate
        cloned = SelfLearningAgent(9, 5, lr=survivor.lr, name=f"{survivor.name}_clone")
        cloned.policy.load_state_dict(survivor.policy.state_dict())
        cloned.mutate()
        new_agents.append(survivor)
        new_agents.append(cloned)

    return new_agents

# ===== Main Runner =====
def main():
    random.seed(42)
    torch.manual_seed(42)

    agents = [SelfLearningAgent(9, 5, name=f"Bot_{i}") for i in range(32)]

    generations = 50

    try:
        for generation in range(1, generations + 1):
            agents = tournament_generation(agents, games_per_agent=5, generation_num=generation)
            avg_mmr = sum(agent.mmr for agent in agents) // len(agents)
            top_agent = max(agents, key=lambda x: x.mmr)

            print(f"\n=== Generation {generation}: Average MMR = {avg_mmr}, Top Agent = {top_agent.name} (MMR: {top_agent.mmr:.0f}) ===")
    except KeyboardInterrupt:
        print("\n⚡ Training interrupted! Saving current agents...")
        for agent in agents:
            agent.save(f"saved_models/{agent.name}_generation_final.pt")
        print("✅ Saved!")

if __name__ == "__main__":
    main()
