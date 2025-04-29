import torch
import torch.nn as nn
import torch.optim as optim

class TinyNNPolicy(nn.Module):
    def __init__(self, input_size, action_size):
        super(TinyNNPolicy, self).__init__()
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

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

import torch

class SelfLearningAgent:
    def __init__(self, input_size, action_size, lr=1e-3, name="Agent"):
        self.name = name
        self.policy = TinyNNPolicy(input_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []
        self.cumulative_reward = 0.0
        self.mmr = 0
        self.behavior_counts = {
            'buy': 0,
            'sell': 0,
            'roll': 0,
            'level': 0,
            'end_turn': 0,
        }


    def act(self, state):
        action, probs = self.policy.predict_action(state)
        log_prob = torch.log(probs[action])
        self.memory.append(log_prob)
        return action

    def learn(self, final_game_reward):
        if len(self.memory) == 0:
            return  # No actions taken, skip learning step

        total_reward = self.cumulative_reward + final_game_reward

        # 🟰 Reward normalization (tiny trick)
        rewards = torch.tensor([total_reward for _ in self.memory])
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # Add small epsilon to avoid divide-by-zero

        loss = []
        for log_prob, reward in zip(self.memory, rewards):
            loss.append(-log_prob * reward)
        loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []  # Reset memory
        self.cumulative_reward = 0.0  # Reset cumulative reward



    def visualize_policy(self, state):
        # Visualize the policy's decision-making (action probabilities)
        action, probs = self.policy.predict_action(state)
        print(f"Action probabilities for {self.name}: {probs}")
        return probs
