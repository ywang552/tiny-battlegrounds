import torch
import torch.nn as nn
import torch.nn.functional as F
import json


with open("data/minion_pool.json", "r") as f:
    minion_data = json.load(f)

    TRIBE_TO_INDEX = {
        "Beast": 0,
        "Mech": 1,
        "Murloc": 2,
        "Demon": 3,
        "Dragon": 4,
        "Elemental": 5,
        "Naga": 6,
        "Quilboar": 7,
        "Pirate": 8,
        "Undead": 9,
        "None": 10  # for no tribe / neutral
    }


    def encode_minion(minion, source_flag, slot_idx=None):
        tribes = torch.zeros(11)

        try:
            # 🔥 If minion is Amalgam-style ("All"), set all tribes = 1
            if "All" in minion.types:
                tribes[:] = 1
            else:
                for tribe in minion.types:
                    key = tribe if tribe is not None else "None"
                    tribes[TRIBE_TO_INDEX[key]] = 1

            base = torch.tensor([
                float(minion.attack),
                float(minion.health),
                float(minion.tier),
            ], dtype=torch.float32)

            slot_feature = torch.tensor(
                [slot_idx / 7.0], dtype=torch.float32
            ) if slot_idx is not None else torch.tensor([0.0])

            out = torch.cat([base, tribes, torch.tensor([source_flag], dtype=torch.float32), slot_feature])
            return out

        except Exception as e:
            print(f"❌ Failed to encode minion: {minion.name}, error: {e}")
            return torch.zeros(16)





class TransformerAgent:
    def __init__(self, name="TransformerAgent", action_size=5, embed_dim=128, num_heads=4, num_layers=2):
        self.name = name
        self.action_size = action_size
        self.embed_dim = embed_dim
        self.memory = []
        self.mmr = 0
        self.opponent_memory = {}  # opponent_id → summary vector
        self.env = None  # will be set later by the environment

        # === Token Embedding Layers ===
        self.state_embed = nn.Linear(3, embed_dim)  # [tier, health, turn]
        self.minion_embed = nn.Linear(16, embed_dim)  # [atk, hp, tier, tribes (11), source_flag, slot_idx]
        self.econ_embed = nn.Linear(4, embed_dim)  # [gold, gold_cap, reroll_cost, upgrade_cost]
        self.tier_projector = nn.Linear(6, embed_dim)  # frozen projection
        self.tier_projector.weight.requires_grad = False
        self.tier_projector.bias.requires_grad = False
        self.opponent_embed = nn.Linear(7, embed_dim)  # summary vector: 6 + opponent_id
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        # === Transformer ===
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=4*embed_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Heads ===
        self.policy_head = nn.Linear(embed_dim, action_size)
        self.value_head = nn.Linear(embed_dim, 1)

        # === Optimizer ===
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def parameters(self):
        return list(self.state_embed.parameters()) + \
               list(self.minion_embed.parameters()) + \
               list(self.econ_embed.parameters()) + \
               list(self.tier_projector.parameters()) + \
               list(self.opponent_embed.parameters()) + \
               list(self.transformer.parameters()) + \
               list(self.policy_head.parameters()) + \
               list(self.value_head.parameters()) + [self.cls_token]

    def build_tokens(self, state_vec, board_minions, shop_minions, econ_vec, tier_vec, opponent_vec=None):
        tokens = []
        tokens.append(self.state_embed(state_vec))
        tokens.append(self.econ_embed(econ_vec))
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

    
    @staticmethod
    def build_token_inputs_from_env(env, player_id, phase="active", **kwargs):
        player = env.agents[player_id]
        opponent_id = env.current_opponent.get(player_id, None)

        # === State token: [tier, health, turn] — same for all phases
        state_vec = torch.tensor([
            player.tier,
            player.health,
            env.turn
        ], dtype=torch.float32)

        if phase == "active":
            # === Econ token: real values during buy phase
            econ_vec = torch.tensor([
                player.gold,
                player.gold_cap,
                1,  # can be replaced with dynamic reroll cost if needed
                env.get_upgrade_cost(player),
            ], dtype=torch.float32)

            # === Shop tokens
            shop_minions = [
                encode_minion(minion, source_flag=0, slot_idx=3 + i)
                for i, minion in enumerate(player.shop)
            ]

        else:  # combat phase
            # === Zeroed econ token (no shop during combat)
            econ_vec = torch.tensor([
                0.0, player.gold_cap, 0.0, 0.0
            ], dtype=torch.float32)

            # === No shop minions during combat
            shop_minions = []

        # === Tier token: 1/0 flags for available tiers
        tier_vec = torch.tensor([
            1 if i < player.tier else 0 for i in range(6)
        ], dtype=torch.float32)

        # === Board minions
        board_minions = [
            encode_minion(minion, source_flag=1, slot_idx=i)
            for i, minion in enumerate(player.board)
        ]

        # === Opponent memory (if available)
        opponent_vec = None
        if hasattr(player, 'opponent_memory') and opponent_id in player.opponent_memory:
            opponent_summary = player.opponent_memory[opponent_id]
            opponent_vec = torch.tensor(opponent_summary, dtype=torch.float32)
        econ_vec = torch.nan_to_num(econ_vec, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "state_vec": state_vec,
            "econ_vec": econ_vec,
            "tier_vec": tier_vec,
            "board_minions": board_minions,
            "shop_minions": shop_minions,
            "opponent_vec": opponent_vec
        }

    
    def build_state(self, env, phase="active", **kwargs):
        token_inputs = self.build_token_inputs_from_env(env, env.agents.index(self), phase, **kwargs)
        return self.build_tokens(**token_inputs)
    

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
            "log_prob": log_prob,
            "value": value,
            "reward": None  # to be filled in observe()
        })

        return action
    def _update_opponent_memory(self, opponent):
        opponent_id = self.env.agents.index(opponent)
        board = opponent.board
        strength = sum(m.strength() for m in board)
        avg_tier = sum(m.tier for m in board) / len(board) if board else 0
        damage = opponent.tier + sum(m.tier for m in board)

        # replace or initialize
        self.opponent_memory[opponent_id] = [
            0,
            strength,
            avg_tier,
            damage,
            0, 
            opponent.tier,
            opponent_id
        ]

        # increment all other memories' turns since seen
        for k in self.opponent_memory:
            if k != opponent_id:
                self.opponent_memory[k][0] += 1



    def observe(self, token_input, reward, opponent=None):
        self.memory.append({
            "input": token_input,
            "reward": reward
        })

        if opponent is not None:
            self._update_opponent_memory(opponent)

    def learn(self, final_mmr, gamma=0.95, lambd=0.1):
        policy_loss = []
        value_loss = []

        for entry in self.memory:
            r = entry["reward"] + lambd * final_mmr
            adv = r - entry["value"]
            policy_loss.append(-entry["log_prob"] * adv.detach())
            value_loss.append(adv.pow(2))

        loss = torch.stack(policy_loss).sum() + 0.5 * torch.stack(value_loss).sum()

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
