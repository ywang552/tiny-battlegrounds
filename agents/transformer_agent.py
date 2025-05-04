import torch
import torch.nn as nn
import torch.nn.functional as F
import json


with open("data/minion_pool.json", "r") as f:
    minion_data = json.load(f)

class TransformerAgent(nn.Module):

    def __init__(self, name="TransformerAgent", action_size=16, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()

        self.name = name
        self.action_size = action_size
        self.embed_dim = embed_dim
        self.memory = []
        self.mmr = 0
        # === Token Embedding Layers ===
        self.state_embed = nn.Linear(3, embed_dim)  # [tier, health, turn]
        self.minion_embed = nn.Linear(15, embed_dim)  # [atk, hp, tier, tribes (10), source_flag, slot_idx]
        self.econ_embed = nn.Linear(6, embed_dim)  # [gold, gold_cap, reroll_cost, upgrade_cost]
        self.tier_projector = nn.Linear(6, embed_dim)  # frozen projection
        self.tier_projector.weight.requires_grad = False
        self.tier_projector.bias.requires_grad = False
        self.opponent_embed = nn.Linear(4, embed_dim)  # summary vector: 6 + opponent_id
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.turns_skipped_this_game = 0 
        self.gold_spent_this_game = 0 
        self.minions_bought_this_game = 0 
        # === Transformer ===
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=4*embed_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Heads ===
        self.policy_head = nn.Linear(embed_dim, action_size)
        self.value_head = nn.Linear(embed_dim, 1)

        # === Optimizer ===
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.behavior_counts = {
            "buy": 0,
            "sell": 0,
            "roll": 0,
            "level": 0,
            "end_turn": 0,
        }


    def parameters(self):
        return list(self.state_embed.parameters()) + \
               list(self.minion_embed.parameters()) + \
               list(self.econ_embed.parameters()) + \
               list(self.tier_projector.parameters()) + \
               list(self.opponent_embed.parameters()) + \
               list(self.transformer.parameters()) + \
               list(self.policy_head.parameters()) + \
               list(self.value_head.parameters()) + [self.cls_token]
    
    def build_tokens(self, state_vec, board_minions, shop_minions, econ_vec, tier_vec, current_turn=None, opponent_vec=None):
        # Add turn progression signal (0-1 normalized)
        turn_progress = torch.tensor([current_turn / 20.0]) if current_turn is not None else torch.tensor([0.0])
        
        # Enhanced econ embedding
        enhanced_econ = torch.cat([
            econ_vec,
            turn_progress,
            torch.tensor([len(board_minions) / 7.0])  # Board commitment
        ])
    
        tokens = []
    

        tokens.append(self.state_embed(state_vec))
        tokens.append(self.econ_embed(enhanced_econ))
        tokens.append(self.tier_projector(tier_vec))



        # ✅ board_minions and shop_minions already include slot index in encode_minion()
        for minion in board_minions:
            tokens.append(self.minion_embed(minion))

        for minion in shop_minions:
            tokens.append(self.minion_embed(minion))

        if opponent_vec is not None:
            tokens.append(self.opponent_embed(opponent_vec))

        cls = self.cls_token.unsqueeze(0)  # [1, D]
        all_tokens = torch.stack(tokens).unsqueeze(0)  # [1, N, D]
        all_tokens = torch.cat([cls, all_tokens], dim=1)  # [1, N+1, D]

        return all_tokens


    def act(self, token_input, action_mask=None):
        output = self.transformer(token_input)  # [1, N+1, embed_dim]
        cls_output = output[0, 0]               # [embed_dim]

        logits = self.policy_head(cls_output)   # [16]
        logits = logits - logits.max()          # stabilizer
        # print(action_mask) ## TODO
        # === Apply action mask (optional) ===
        if action_mask is not None:
            logits[~action_mask] = -float("inf")

            ## TODO
            # if action_mask.sum() == 0:
            #     print("⚠️ No valid actions — forcing end_turn fallback")
            #     logits[:] = -float("inf")
            #     logits[self.END_TURN_IDX] = 0.0

            # # 🔍 Debugging info
            # print("🎯 Action Mask:", action_mask.tolist())
            # print("📈 Masked Logits:", logits.tolist())



        probs = F.softmax(logits, dim=-1)

        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            print("🚨 Bad probs triggered!")
            print("  CLS output:", cls_output)
            print("  Logits:", logits)
            print("  Probs:", probs)
            print("  Agent name:", self.name)
            print("  Action size:", self.action_size)
            print("  Mask:", action_mask)

        action = torch.multinomial(probs, 1).item()
        value = self.value_head(cls_output).squeeze()
        log_prob = torch.log(probs[action])

        self.memory.append({
            "state": token_input,
            "log_prob": log_prob,
            "value": value,
            "reward": None,
        })

        return action


    def observe(self, state, reward, turn=None):
        self.current_turn = turn  # NEW: Store turn when observing
    # ... rest of original code ...
        """Ensure all entries have value estimates"""
        with torch.no_grad():
            # Generate value estimate for all observations
            output = self.transformer(state)
            cls_output = output[0, 0]
            value = self.value_head(cls_output).squeeze()
        
        self.memory.append({
            "state": state,
            "reward": reward,
            "value": value  # Add value estimate to all entries
        })

    def learn(self, reward):
        if not self.memory:
            return
            
        # Calculate advantage using the reward (MMR)
        advantages = []
        for entry in self.memory:
            value = entry.get('value', 0)
            advantage = reward - value.item() if torch.is_tensor(value) else reward - value
            advantages.append(advantage)
        
        # Normalize advantages
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate losses
        policy_loss = []
        value_loss = []
        for entry, advantage in zip(self.memory, advantages):
            if 'log_prob' in entry:
                policy_loss.append(-entry['log_prob'] * advantage)
            if 'value' in entry:
                value_loss.append(F.mse_loss(entry['value'], torch.tensor(reward, dtype=torch.float32)))

        
        # Update if we have any losses
        if policy_loss or value_loss:
            loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
        
        self.memory.clear()





    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))
        
    def state_dict(self):
        return {
            # "state_embed": self.state_embed.state_dict(),
            # "minion_embed": self.minion_embed.state_dict(),
            # "econ_embed": self.econ_embed.state_dict(),
            # "tier_projector": self.tier_projector.state_dict(),
            # "opponent_embed": self.opponent_embed.state_dict(),
            # "cls_token": self.cls_token.data,
            "transformer": self.transformer.state_dict(),
            "policy_head": self.policy_head.state_dict(),
            "value_head": self.value_head.state_dict(),
        }

    def load_state_dict(self, state_dict):
        # self.state_embed.load_state_dict(state_dict["state_embed"])
        # self.minion_embed.load_state_dict(state_dict["minion_embed"])
        # self.econ_embed.load_state_dict(state_dict["econ_embed"])
        # self.tier_projector.load_state_dict(state_dict["tier_projector"])
        # self.opponent_embed.load_state_dict(state_dict["opponent_embed"])
        # self.cls_token.data.copy_(state_dict["cls_token"])
        self.transformer.load_state_dict(state_dict["transformer"])
        self.policy_head.load_state_dict(state_dict["policy_head"])
        self.value_head.load_state_dict(state_dict["value_head"])

