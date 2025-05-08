import torch
import torch.nn as nn
import torch.nn.functional as F
import json

import os
import logging
LOG_DIR = "agent_logs"
os.makedirs(LOG_DIR, exist_ok=True)


with open("data/minion_pool.json", "r") as f:
    minion_data = json.load(f)

class TransformerAgent(nn.Module):

    def __init__(self, name="TransformerAgent", action_size=16, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.phase_embed = nn.Linear(1, embed_dim)
        

        self.name = name


        self.action_size = action_size
        self.embed_dim = embed_dim
        self.memory = []
        self.mmr = 0
        # === Token Embedding Layers ===
        self.state_embed = nn.Linear(3, embed_dim)  # [tier, health, turn]
        self.minion_embed = nn.Linear(16, embed_dim)  # [atk, hp, tier, tribes (10), source_flag, slot_idx]
        self.econ_embed = nn.Linear(6, embed_dim)  # [gold, gold_cap, reroll_cost, upgrade_cost]
        self.tier_projector = nn.Linear(6, embed_dim)  # frozen projection
        self.tier_projector.weight.requires_grad = False
        self.tier_projector.bias.requires_grad = False
        self.opponent_embed = nn.Linear(5, embed_dim)  
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.combat_embed = nn.Linear(2, embed_dim)

        self.turns_skipped_this_game = 0 
        self.gold_spent_this_game = 0 
        self.minions_bought_this_game = 0 
        # === Transformer ===
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=4*embed_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Heads ===
        self.policy_head = nn.Linear(embed_dim, action_size)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Tanh()
        )
        
        # === Optimizer ===
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.behavior_counts = {
            "buy": 0,
            "sell": 0,
            "roll": 0,
            "level": 0,
            "end_turn": 0,
        }

        # === Add logger for this agent ===
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)

        log_path = os.path.join(LOG_DIR, f"{self.name}.log")
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')

        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)

        # Avoid duplicated handlers
        if not self.logger.handlers:
            self.logger.addHandler(fh)



    def parameters(self):
        return list(self.state_embed.parameters()) + \
               list(self.minion_embed.parameters()) + \
               list(self.econ_embed.parameters()) + \
               list(self.tier_projector.parameters()) + \
               list(self.opponent_embed.parameters()) + \
               list(self.transformer.parameters()) + \
               list(self.policy_head.parameters()) + \
               list(self.value_head.parameters()) + [self.cls_token]
    
    def build_attention_mask(self, phase, board_minions, shop_minions, combat_vec=None):
        attention_mask = [1]  # [CLS]

        # Core tokens
        attention_mask.append(1)  # state_vec
        attention_mask.append(1 if phase == "active" else 0)  # econ_vec
        attention_mask.append(1 if phase == "active" else 0)  # tier_vec


        # Board minions
        for minion in board_minions:
            is_padding = torch.all(minion[:3] == -1)
            attention_mask.append(0 if is_padding else 1)

        # Shop minions
        if phase == "active":
            for minion in shop_minions:
                is_padding = torch.all(minion == 0)
                attention_mask.append(0 if is_padding else 1)
        else:
            attention_mask += [0] * len(shop_minions)

        # Opponent summary
        attention_mask.append(1 if phase == "active" else 0)

        # Combat summary
        attention_mask.append(1 if phase == "combat" and combat_vec is not None else 0)

        return torch.tensor(attention_mask).unsqueeze(0)  # [1, seq_len]


    def build_tokens(
        self,
        state_vec,
        board_minions,
        shop_minions,
        econ_vec,
        tier_vec,
        opponent_vec,
        combat_vec,
        phase="active"
    ):
        current_turn = state_vec[2].item()
        turn_progress = torch.tensor([current_turn / 20.0])
        enhanced_econ = torch.cat([
            econ_vec,
            turn_progress,
            torch.tensor([len(board_minions) / 7])
        ])

        tokens = []

        # 1. CLS
        tokens.append(self.cls_token.squeeze(0))  # [D]

        # 2. Core state
        tokens.append(self.state_embed(state_vec))          # [D]
        tokens.append(self.econ_embed(enhanced_econ))       # [D]
        tokens.append(self.tier_projector(tier_vec))        # [D]

        # 3. Board
        for minion in board_minions:
            tokens.append(self.minion_embed(minion))        # [D] x7

        # 4. Shop
        for minion in shop_minions:
            tokens.append(self.minion_embed(minion))        # [D] x6

        # 5. Opponent
        tokens.append(self.opponent_embed(opponent_vec))    # [D]

        # 6. Combat
        tokens.append(self.combat_embed(combat_vec))        # [D]

        # 7. Stack
        token_tensor = torch.stack(tokens).unsqueeze(0)     # [1, 19, D]

        # 8. Attention Mask
        attention_mask = self.build_attention_mask(phase, board_minions, shop_minions)

        # 9. Sanity Check
        assert token_tensor.shape[1] == attention_mask.shape[1] == 19, \
            f"Shape mismatch: tokens={token_tensor}, mask={attention_mask}"

        return token_tensor, attention_mask




    def act(self, token_input, action_mask=None, attention_mask=None):
        # === Run Transformer with attention mask ===
        transformer_output = self.transformer(
            token_input,  # shape: [1, seq_len, D]
            src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None
        )  # shape: [1, seq_len, D]

        # === Use CLS token output ===
        cls_output = transformer_output[0, 0]  # shape: [D]

        # === Policy head ===
        logits = self.policy_head(cls_output)  # shape: [action_size]
        logits = logits - logits.max()         # numerical stabilizer

        # === Apply action mask ===
        if action_mask is not None:
            logits[~action_mask] = -float("inf")
            if action_mask.sum() == 0:
                logits[:] = -float("inf")
                logits[self.END_TURN_IDX] = 0.0
                print("⚠️ No valid actions - defaulting to end_turn")

        # === Softmax to get probabilities ===
        probs = F.softmax(logits, dim=-1)

        # === Fallback if probs are invalid ===
        if not torch.isfinite(probs).all():
            probs = torch.ones_like(probs) / len(probs)
            print("🚨 Invalid probs - using uniform distribution")

        # === Log for rare numerical issues ===
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            print("🚨 Bad probs triggered!")
            print("  CLS output:", cls_output)
            print("  Logits:", logits)
            print("  Probs:", probs)
            print("  Agent name:", self.name)
            print("  Action size:", self.action_size)
            print("  Mask:", action_mask)

        # === Sample action ===
        action = torch.multinomial(probs, 1).item()
        value = self.value_head(cls_output).squeeze()
        log_prob = torch.log(probs[action])

        if not hasattr(self, "debug_turn_counter"):
            self.debug_turn_counter = 0

        if self.debug_turn_counter == 0:
            self.logger.info(f"\nTurn Start")
            self.logger.info(f"  Action Mask: {action_mask.int().tolist()}")
            self.logger.info(f"  Logits: {[round(x, 2) for x in logits.detach().cpu().tolist()]}")
            self.logger.info(f"  Probs: {[round(x, 3) for x in probs.detach().cpu().tolist()]}")
            self.logger.info(f"  Chosen Action: {action}")

        self.debug_turn_counter += 1

        if action == 15:
            self.logger.info(f"  ⏹️ End Turn ({self.debug_turn_counter} decisions)\n")
            self.debug_turn_counter = 0





        # === Store in memory ===
        self.memory.append({
            "state": token_input,
            "log_prob": log_prob,
            "value": value,
            "reward": None,
        })

        return action



    def observe(self, token_input, reward, attention_mask=None):

        with torch.no_grad():
            output = self.transformer(
                token_input,
                src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None
            )
            cls_output = output[0, 0]  # extract CLS embedding
            value = self.value_head(cls_output).squeeze()  # scalar

        self.memory.append({
            "state": token_input,
            "reward": reward,
            "value": value
        })


    def learn(self, reward: float):
        if not self.memory:
            return  # Nothing to learn from
        self.value_errors = []
        self.value_preds = []


        # === 1. Compute advantage for each memory entry ===
        advantages = []
        for entry in self.memory:
            value = entry.get('value', 0.0)
            if isinstance(value, torch.Tensor):
                value = value.item()
            advantage = reward - value
            advantages.append(advantage)

        # Normalize advantages (helps training stability)
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === 2. Compute policy and value loss ===
        policy_losses = []
        value_losses = []

        for entry, adv in zip(self.memory, advantages):
            # --- Train the policy head ---
            log_prob = entry.get("log_prob", None)
            if log_prob is not None:
                policy_losses.append(-log_prob * adv)

            # --- Train the value head ---
            value = entry.get("value", None)
            if value is not None:
                target = torch.tensor(reward, dtype=torch.float32)
                loss = F.mse_loss(value, target)
                value_losses.append(loss)
            
            if self.name == "Transformer_0":
                self.value_errors.append(loss.detach().item())
                self.value_preds.append(value.detach().item() if torch.is_tensor(value) else value)

        # === 3. Backpropagation ===
        if policy_losses or value_losses:
            total_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

            self.optimizer.zero_grad()        # Reset gradients
            total_loss.backward()             # Compute gradients through:
                                            # ✅ value head
                                            # ✅ policy head
                                            # ✅ transformer
                                            # ✅ input embeddings
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)  # Stability
            self.optimizer.step()             # Apply updates


        # === Diagnostics ===
        try:
            avg_value = torch.stack([e['value'] for e in self.memory if 'value' in e]).mean().item()
            avg_logprob = torch.stack([e['log_prob'] for e in self.memory if 'log_prob' in e]).mean().item()
            ploss = torch.stack(policy_losses).sum().item() if policy_losses else 0.0
            vloss = torch.stack(value_losses).sum().item() if value_losses else 0.0
        except Exception as e:
            avg_value, avg_logprob, ploss, vloss = 0, 0, 0, 0
            self.logger.warning(f"[Warning] Failed to compute diagnostics in learn(): {e}")

        # === Log
        self.logger.info(f"\n🎓 Learn Summary")
        self.logger.info(f"  Reward: {reward:.2f}")
        self.logger.info(f"  Avg Value Estimate: {avg_value:.3f}")
        self.logger.info(f"  Avg LogProb: {avg_logprob:.3f}")
        self.logger.info(f"  Policy Loss: {ploss:.4f}")
        self.logger.info(f"  Value Loss: {vloss:.4f}")
        self.logger.info(f"🎓 End of Game for {self.name}\n")


        # === 4. Clear memory ===
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

