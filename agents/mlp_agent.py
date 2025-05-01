import torch
import torch.nn as nn
import torch.optim as optim
import random


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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SelfLearningAgent:
    def __init__(self, input_size, action_size, lr=1e-3, name="Agent", ancestor=None):
        self.name = name
        self.policy = TinyNNPolicy(input_size, action_size)
        self.value_net = TinyValueNet(input_size)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()), lr=lr
        )
        self.lr = lr
        self.memory = []
        self.mmr = 0.0
        self.ancestor = ancestor if ancestor is not None else name

    def act(self, state):
        action, probs = self.policy.predict_action(state)
        log_prob = torch.log(probs[action])
        value = self.value_net(state)
        self.memory.append({
            "state": state.detach(),
            "action": action,
            "log_prob": log_prob,
            "value": value,
        })
        return action

    def observe(self, next_state, reward, opponent=None):
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
            s, s_, a = entry["state"], entry["next_state"], entry["action"]
            log_prob, value = entry["log_prob"], entry["value"]

            with torch.no_grad():
                next_value = self.value_net(s_)

            advantage = (r + gamma * next_value) - value
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

    def build_state(self, env, phase="active", **kwargs):
        if phase == "combat":
            return env.build_combat_state(self, kwargs["previous_health"], kwargs["enemy_strength"])
        else:
            return env.build_active_state(self)
