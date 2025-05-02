import torch
import torch.nn as nn
import torch.nn.functional as F
import json


with open("data/minion_pool.json", "r") as f:
    minion_data = json.load(f)





class TransformerAgent:

    def __init__(self, name="TransformerAgent", action_size=5, embed_dim=128, num_heads=4, num_layers=2):
        self.name = name
        self.action_size = action_size
        self.embed_dim = embed_dim
        self.memory = []
        self.mmr = 0
        self.opponent_memory = {}  # opponent_id → summary vector
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
    
    def _update_opponent_memory(self, opponent, turn):
        if opponent is None:
            return

        if not hasattr(self, "opponent_memory"):
            self.opponent_memory = {}

        # Always update with fresh summary
        strength = sum(m.attack + m.health for m in opponent.board)
        summary = [
            float(opponent.tier),
            float(strength),
            0.0,  # hp_delta only known at build_state time
            float(turn) 
        ]
        self.opponent_memory[opponent] = summary


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


    def act(self, token_input):

        output = self.transformer(token_input)  # [1, N+1, embed_dim]
        cls_output = output[0, 0]  # [embed_dim]

        logits = self.policy_head(cls_output)
        logits = logits - logits.max()  # stabilizer
        probs = F.softmax(logits, dim=-1)

        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            print("🚨 Bad probs triggered!")
            print("  CLS output:", cls_output)
            print("  Logits:", logits)
            print("  Probs:", probs)
            print("  Agent name:", self.name)
            print("  Action size:", self.action_size)


        action = torch.multinomial(probs, 1).item()


        value = self.value_head(cls_output).squeeze()

        log_prob = torch.log(probs[action])

        self.memory.append({
            "state": token_input,  # Add state reference
            "log_prob": log_prob,
            "value": value,
            "reward": None,  # To be filled later
            "turn": getattr(self, 'env', None).turn
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
            "turn": turn if turn is not None else getattr(self, 'env', None).turn,
            "value": value  # Add value estimate to all entries
        })

    def learn(self, final_mmr):
        """Handle all memory entry formats safely"""
        if not self.memory:
            return
            
        # Get final turn safely
        final_turn = max(entry.get("turn", 1) for entry in self.memory)
        
        # Process all entries
        policy_loss = []
        value_loss = []
        
        for entry in self.memory:
            # Skip entries missing required fields
            if "value" not in entry:
                continue
                
            # Calculate reward weighting
            turn_weight = entry.get("turn", 1) / final_turn
            reward = final_mmr * turn_weight
            
            # Handle both action and observation entries
            if "log_prob" in entry:  # Action entry
                advantage = reward - entry["value"]
                policy_loss.append(-entry["log_prob"] * advantage.detach())
                value_loss.append(advantage.pow(2))
            else:  # Observation entry
                value_loss.append((reward - entry["value"]).pow(2))
        
        if policy_loss or value_loss:
            loss = (torch.stack(policy_loss).sum() if policy_loss else 0) + \
                (0.5 * torch.stack(value_loss).sum() if value_loss else 0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.memory.clear()



    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))
        
    def state_dict(self):
        return {
            "policy_head": self.policy_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "transformer": self.transformer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.policy_head.load_state_dict(state_dict["policy_head"])
        self.value_head.load_state_dict(state_dict["value_head"])
        self.transformer.load_state_dict(state_dict["transformer"])
