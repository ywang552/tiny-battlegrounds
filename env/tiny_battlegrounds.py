import random
import torch
from utils.load import load_minions
from agents.transformer_agent import TransformerAgent
class Minion:
    def __init__(self, name, types, attack, health, tier, keywords=None):
        self.name = name
        self.types = types
        self.attack = attack
        self.health = health
        self.tier = tier
        self.keywords = set(keywords or [])

    def strength(self):
        return self.attack + self.health

    def __repr__(self):
        return f"{self.name}({self.attack}/{self.health})"

path = "data\\bg_minions_all.json"  # your full file
MINION_POOL = [Minion(**data) for data in load_minions(path)]


# --- TinyBattlegrounds Environment ---
class TinyBattlegroundsEnv:
    # === GAME CONSTANTS ===
    MINION_COST = 3
    ROLL_COST = 1
    MAX_BOARD_SIZE = 7
    MAX_SHOP_SIZE = 6
    INITIAL_GOLD = 3
    MAX_TIER = 6

        # === Action Space Index Mapping ===
    BUY_START = 0
    BUY_END = 5           # slots 0 to 5 (inclusive)

    SELL_START = 6
    SELL_END = 12         # slots 6 to 12 (inclusive)

    ROLL_IDX = 13
    LEVEL_IDX = 14
    END_TURN_IDX = 15

    ACTION_SIZE = 16      # total number of actions


    SHOP_SLOTS = {
        1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 6
    }

    TAVERN_UPGRADE_COST = {
        1: 5, 2: 7, 3: 8, 4: 9, 5: 10
    }

    PLACEMENT_REWARDS = {
        1: 1, 2: .75, 3: .5, 4: .25,
        5: -.25, 6: -.5, 7: -.75, 8: -1
    }

    @staticmethod
    def encode_minion(minion, source_flag, slot_idx=None):
        # More efficient tribe encoding
        TRIBE_TO_INDEX = {
            "Beast": 0, "Mech": 1, "Murloc": 2, "Demon": 3,
            "Dragon": 4, "Elemental": 5, "Naga": 6, "Quilboar": 7,
            "Pirate": 8, "Undead": 9, "None": 10
        }

        # Initialize with zeros (more memory efficient)
        tribes = torch.zeros(11)  # Now includes "None" explicitly
        base = torch.full((3,), -1.0)  # [attack, health, tier]


        if minion is not None:
            # Handle tribes more efficiently
            types = minion.types or ["None"]
            if "All" in types:
                tribes[:10] = 1  # Set all tribes to 1 except "None"
            else:
                for tribe in types:
                    idx = TRIBE_TO_INDEX.get(tribe, 10)  # Default to "None"
                    tribes[idx] = 1

            base = torch.tensor([
                float(minion.attack),
                float(minion.health),
                float(minion.tier),
            ], dtype=torch.float32)

        # More robust slot feature handling
        max_slot = 5.0 if source_flag == 0 else 6.0  # shop vs board
        slot_feature = torch.tensor(
            [slot_idx / max_slot] if slot_idx is not None else [0.0],
            dtype=torch.float32
        )

        return torch.cat([
            base,                          # [3]
            tribes,                        # [11]
            torch.tensor([source_flag]),    # [1]
            slot_feature                   # [1]
        ])  # Total dim: 16

    def __init__(self, agents):
        self.agents = agents
        self.turn = 1
        self.dead = {}  # {agent: turn number}
        self.latest_dead_agent = None
        self.setup()

    def setup(self):
        for agent in self.agents:
            agent.gold_cap = self.INITIAL_GOLD
            agent.gold = self.INITIAL_GOLD
            agent.tier = 1
            agent.board = []
            agent.health = 40
            agent.alive = True
            agent.opponent_memory = {}  # opponent_id → summary vector
            agent.shop = self.roll_shop(agent.tier)
            agent.gold_spent_this_game = 0
            agent.gold_spent_this_turn = 0
            agent.gold_earned_this_turn = 0
            agent.minions_bought_this_game = 0
            agent.turns_skipped_this_game = 0
            agent.behavior_counts = {'buy': 0, 'sell': 0, 'roll': 0, 'level': 0, 'end_turn': 0}
            agent.tavern_upgrade_cost = self.TAVERN_UPGRADE_COST.get(agent.tier, -1)

    def step(self, verbose=False, focus_agent_name=None):
        self._prepare_matchups()
        self._run_active_phase(verbose, focus_agent_name)
        self._run_combat_phase()
        self.remove_dead()

    def _prepare_matchups(self):
        alive_agents = [a for a in self.agents if a.alive]
        self.match_pairs = self.matchmaking(alive_agents)
        self.current_opponent = {}

        for p1, p2 in self.match_pairs:
            if p2 is None:
                self.current_opponent[p1] = None
            else:
                self.current_opponent[p1] = p2
                self.current_opponent[p2] = p1

    def _run_active_phase(self, verbose, focus_agent_name):
        for agent in self.agents:
            if not agent.alive:
                continue
            self._setup_agent_turn(agent, verbose, focus_agent_name)
            self._simulate_agent_turn(agent, verbose, focus_agent_name)
        
        self._reduce_upgrade_costs(verbose, focus_agent_name)

    def _setup_agent_turn(self, agent, verbose=False, focus_agent_name=None):
        # Refresh gold cap based on turn number
        agent.gold_cap = min(self.INITIAL_GOLD + self.turn - 1, 10)
        agent.gold = agent.gold_cap

        agent.shop = self.roll_shop(agent.tier)
        
        ## TODO
        if verbose and agent.name == focus_agent_name and agent.alive:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Agent_({focus_agent_name})-> turn: {self.turn}, gold: {agent.gold}, tavern: {agent.tier}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def _run_combat_phase(self):
        for p1, p2 in self.match_pairs:
            self.simulate_combat(p1, p2)

    def _simulate_agent_turn(self, agent, verbose=False, focus_agent_name=None):

        while agent.gold >= 0:

            state = self.build_state(agent, phase="active")
            action = agent.act(
                token_input=state["tokens"],
                action_mask=state["masks"]["action"],                      # ✅ your gold/board/shop logic
                attention_mask=state["masks"]["attention"]  # ✅ internal sequence visibility
            )

            action_str = self.decode_action(agent, action)



            self._record_behavior(agent, action_str)
            if verbose and agent.name == focus_agent_name:
                print(f"Action chosen: {action_str}")

            if action_str == "end_turn":
                self._handle_end_turn(agent, verbose, focus_agent_name)
                break
            elif action_str == "level":
                self._handle_level(agent, verbose, focus_agent_name)
            elif action_str.startswith("buy_"):
                self._handle_buy(agent, int(action_str.split("_")[1]), verbose, focus_agent_name)
            elif action_str == "roll":
                self._handle_roll(agent)
            elif action_str.startswith("sell_"):
                self._handle_sell(agent, int(action_str.split("_")[1]), verbose, focus_agent_name)

    def _record_behavior(self, agent, action_str):
        if hasattr(agent, "behavior_counts"):
            for key in agent.behavior_counts:
                if action_str.startswith(key):
                    agent.behavior_counts[key] += 1

    def _handle_end_turn(self, agent, verbose, focus_agent_name):
        if agent.gold > 5:
            agent.turns_skipped_this_game += 1
        if verbose and agent.name == focus_agent_name and agent.gold > 5:
            print(">> Ending turn early.")
        state = self.build_state(agent, phase="active")
        agent.observe(state["tokens"], 0.0, attention_mask=state["masks"]["attention"])

    def _handle_level(self, agent, verbose, focus_agent_name):

        cost = agent.tavern_upgrade_cost
        if agent.gold >= cost:
            agent.gold -= cost
            agent.tier += 1
            agent.tavern_upgrade_cost = self.TAVERN_UPGRADE_COST.get(agent.tier, -1)
            agent.gold_spent_this_game += cost
            agent.gold_spent_this_turn += cost
            if verbose and agent.name == focus_agent_name:
                print(f">> Upgraded to Tier {agent.tier}")
        state = self.build_state(agent, phase="active")
        agent.observe(state["tokens"], 0.0, attention_mask=state["masks"]["attention"])

    def _handle_buy(self, agent, idx, verbose, focus_agent_name):
        if agent.shop and idx < len(agent.shop) and agent.gold >= self.MINION_COST and len(agent.board) < self.MAX_BOARD_SIZE:
            bought_minion = agent.shop[idx]
            agent.gold -= self.MINION_COST
            agent.board.append(agent.shop.pop(idx))
            agent.minions_bought_this_game += 1
            agent.gold_spent_this_turn += self.MINION_COST
            agent.gold_spent_this_game += self.MINION_COST
            if verbose and agent.name == focus_agent_name:
                print(f">> Bought {bought_minion.name} ({bought_minion.attack}/{bought_minion.health})")
        state = self.build_state(agent, phase="active")
        agent.observe(state["tokens"], 0.0, attention_mask=state["masks"]["attention"])

    def _handle_roll(self, agent):
        if agent.gold >= self.ROLL_COST:
            agent.gold -= self.ROLL_COST
            agent.shop = self.roll_shop(agent.tier)
            agent.gold_spent_this_game += 1
            agent.gold_spent_this_turn += 1
        state = self.build_state(agent, phase="active")
        agent.observe(state["tokens"], 0.0, attention_mask=state["masks"]["attention"])

    def _handle_sell(self, agent, idx, verbose, focus_agent_name):
        if idx < len(agent.board):
            sold_minion = agent.board.pop(idx)
            agent.gold += 1
            if verbose and agent.name == focus_agent_name:
                print(f">> Sold {sold_minion.name} ({sold_minion.attack}/{sold_minion.health})")
            agent.gold_earned_this_turn += 1
        state = self.build_state(agent, phase="active")
        agent.observe(state["tokens"], 0.0, attention_mask=state["masks"]["attention"])

    def _reduce_upgrade_costs(self, verbose=False, focus_agent_name=None):
        for agent in self.agents:
            old_cost = agent.tavern_upgrade_cost
            agent.tavern_upgrade_cost = max(old_cost - 1, 0)

            if verbose and agent.name == focus_agent_name and agent.alive:

                print(f"[Turn {self.turn}] {agent.name} (Tier {agent.tier}) upgrade cost: {old_cost} → {agent.tavern_upgrade_cost}")
                print(agent.behavior_counts)
                print(f"[Turn {self.turn}] {agent.name} spent:{agent.gold_spent_this_turn}, earned:{agent.gold_earned_this_turn}, start with: {agent.gold_cap}, remaining: {agent.gold}, net: {agent.gold_spent_this_turn + agent.gold - agent.gold_earned_this_turn - agent.gold_cap}")

            agent.gold_spent_this_turn = 0 
            agent.gold_earned_this_turn = 0
            agent.behavior_counts = {'buy': 0, 'sell': 0, 'roll': 0, 'level': 0, 'end_turn': 0}

    def matchmaking(self, alive_agents):
        pairs = []
        random.shuffle(alive_agents)
        if len(alive_agents) % 2 == 1:
            ghost_fighter = alive_agents.pop()
            assert self.latest_dead_agent is not None, "Ghost pairing requested before any agent has died"
            ghost_opponent = self.latest_dead_agent
            pairs.append((ghost_fighter, ghost_opponent))


        for i in range(0, len(alive_agents), 2):
            pairs.append((alive_agents[i], alive_agents[i+1]))

        return pairs

    def roll_shop(self, tier):
        pool = [m for m in MINION_POOL if m.tier <= tier]
        slots = self.SHOP_SLOTS.get(tier, -1)
        return random.sample(pool, min(slots, len(pool)))

    def get_action_mask(self, agent):

        mask = torch.zeros(16, dtype=torch.bool)

        for i in range(self.BUY_START, self.BUY_END + 1):
            slot = i - self.BUY_START
            if slot < len(agent.shop) and agent.gold >= self.MINION_COST and len(agent.board) < self.MAX_BOARD_SIZE:
                mask[i] = True

        for i in range(self.SELL_START, self.SELL_END + 1):
            slot = i - self.SELL_START
            if slot < len(agent.board):
                mask[i] = True

        if agent.gold >= self.ROLL_COST:
            mask[self.ROLL_IDX] = True

        if agent.tier < self.MAX_TIER and agent.gold >= agent.tavern_upgrade_cost:
            mask[self.LEVEL_IDX] = True

        mask[self.END_TURN_IDX] = True

        return mask

    def decode_action(self, agent, action_idx):
        if self.BUY_START <= action_idx <= self.BUY_END:
            idx = action_idx - self.BUY_START
            return f"buy_{idx}"
        elif self.SELL_START <= action_idx <= self.SELL_END:
            idx = action_idx - self.SELL_START
            return f"sell_{idx}"
        elif action_idx == self.ROLL_IDX:
            return "roll"
        elif action_idx == self.LEVEL_IDX:
            return f"level"

        elif action_idx == self.END_TURN_IDX:
            return "end_turn"

        return "invalid"



    def simulate_combat(self, a, d):
        m1 = sum(m.strength() for m in a.board)
        m2 = sum(m.strength() for m in d.board)

        # Tie case
        if m1 == m2:
            if a.alive:
                state = self.build_state(a, phase="combat", hp_delta=0.0, opponent_strength=m2)
                a.observe(state["tokens"], 0.0, attention_mask=state["masks"]["attention"])
            if d.alive:
                state = self.build_state(a, phase="combat", hp_delta=0.0, opponent_strength=m1)
                d.observe(state["tokens"], 0.0, attention_mask=state["masks"]["attention"])
            return

        # Determine winner and loser
        winner = a if m1 > m2 else d
        loser  = d if m1 > m2 else a

        # Winner reward
        if winner.alive:
            state = self.build_state(winner, phase="combat", hp_delta=0.0, opponent_strength=sum(m.strength() for m in loser.board))
            winner.observe(state["tokens"], 0, attention_mask=state["masks"]["attention"])
            

        # Loser punishment
        if loser.alive:
            damage = winner.tier + sum(m.tier for m in winner.board)
            loser.health -= damage
            state = self.build_state(loser, phase="combat", hp_delta=damage, opponent_strength=sum(m.strength() for m in winner.board))
            loser.observe(state["tokens"], 0, attention_mask=state["masks"]["attention"])
            

    def remove_dead(self):
        latest_turn = self.turn
        highest_health_before_death = float('-inf')
        best_candidate = None  # ✅ FIXED

        for agent in self.agents:
            if agent.alive and agent.health <= 0:
                prev_health = agent.health  # already <= 0
                agent.health = max(agent.health, 0)
                agent.alive = False
                self.dead[agent] = latest_turn

                # ✅ Tie-breaker logic:
                if prev_health > highest_health_before_death:
                    best_candidate = agent
                    highest_health_before_death = prev_health

        # ✅ Store best candidate as ghost agent
        if best_candidate:
            self.latest_dead_agent = best_candidate

    def _mark_survivors_as_dead(self):
        for agent in self.agents:
            if agent.alive and agent not in self.dead:
                self.dead[agent] = self.turn

    def _clamp_mmr(self):
        for agent in self.agents:
            agent.mmr = max(0, agent.mmr)

    def play_game(self, verbose=False, focus_agent_name=None):
        while sum(1 for agent in self.agents if agent.alive) > 1:
            self.step(verbose=verbose, focus_agent_name=focus_agent_name)
            self.turn += 1

        self._mark_survivors_as_dead()
        self._clamp_mmr()

        return self.calculate_rewards()

    def _group_agents_by_death_turn(self, ordered_deaths):
        grouped = []
        idx = 0
        while idx < len(ordered_deaths):
            group = [ordered_deaths[idx]]
            idx += 1
            while idx < len(ordered_deaths) and ordered_deaths[idx][1] == group[0][1]:
                group.append(ordered_deaths[idx])
                idx += 1
            grouped.append(group)
        return grouped

    def calculate_rewards(self):
        ordered_deaths = sorted(self.dead.items(), key=lambda x: (x[1], random.random()))
        placement_groups = self._group_agents_by_death_turn(ordered_deaths)

        rewards = {}
        current_place = len(self.agents)

        for group in placement_groups:
            size = len(group)
            reward_sum = sum(self.PLACEMENT_REWARDS.get(current_place - i, -1) for i in range(size))
            average = reward_sum / size

            for agent, _ in group:
                rewards[agent.name] = average
                agent.mmr += average
                agent.mmr = max(0, agent.mmr)

            current_place -= size

        return rewards

    def _get_enemy_summary(self, agent, opponent):
        if opponent in agent.opponent_memory:
            last_turn, last_strength, old_tier = agent.opponent_memory[opponent]
            current_tier = opponent.tier
            alive = float(getattr(opponent, 'alive', 1.0))  # More robust attribute check
            return torch.tensor([
                last_turn, 
                last_strength, 
                old_tier,
                current_tier,
                alive
            ], dtype=torch.float32)
        return torch.zeros(5)  # Default: no enemy data

    def _update_opponent_memory(self, agent, combat_vec):
        opponent = self.current_opponent.get(agent)
        assert opponent, f"Invalid Opponent! Dictionary: {self.current_opponent}, Agent: {agent.name}"
        agent.opponent_memory[opponent] = (self.turn, combat_vec[1], opponent.tier)  # (turn, strength)

    def build_transformer_state(self, agent, phase="active", **kwargs):
        # === 1. Phase Validation ===
        assert phase in ["active", "combat"], f"Invalid Phase: {phase}"
    
        econ_vec = torch.zeros(4)
        opponent_vec = torch.zeros(5)
        combat_vec = torch.zeros(2)
        # === 2. Basic State ===
        state_vec = torch.tensor([
            float(agent.health), 
            float(agent.tier), 
            float(self.turn)
        ], dtype=torch.float32)
        
        # === 3. Phase-Specific Logic ===
        if phase == "active":
            econ_vec = torch.tensor([
                float(agent.gold),
                float(agent.gold_cap),
                1.0,  # Roll cost
                float(agent.tavern_upgrade_cost)
            ], dtype=torch.float32)
            
            opponent = self.current_opponent.get(agent)
            opponent_vec = self._get_enemy_summary(agent, opponent) if opponent else torch.zeros(5)
        else:
            combat_vec = torch.tensor([
                float(kwargs.get("hp_delta", 0.0)),
                float(kwargs.get("enemy_strength", 0.0))
            ], dtype=torch.float32)
            
            opponent = self.current_opponent.get(agent)
            if opponent:
                self._update_opponent_memory(agent, combat_vec)

        
        # === 4. Minion Encoding with Padding ===
        def pad_sequence(sequence, max_len, source_flag):
            encoded = [self.encode_minion(m, source_flag, i) for i, m in enumerate(sequence)]
            padding = [self.encode_minion(None, source_flag) for _ in range(max_len - len(sequence))]
            return torch.stack(encoded + padding)
        
        board_minions = pad_sequence(agent.board, self.MAX_BOARD_SIZE, 1)  # 7 board slots
        shop_minions = pad_sequence(agent.shop, self.MAX_SHOP_SIZE, 0) if phase == "active" else torch.zeros(self.MAX_SHOP_SIZE, 16)
        
        # === Tier vector ===
        tier_vec = torch.tensor([
            1 if i < agent.tier else 0 for i in range(self.MAX_TIER)
        ], dtype=torch.float32)

        tokens, attention_mask = agent.build_tokens(
            state_vec=state_vec,
            board_minions=board_minions,
            shop_minions=shop_minions,
            econ_vec=econ_vec,
            tier_vec=tier_vec,
            opponent_vec=opponent_vec,
            combat_vec=combat_vec,
            phase=phase
        )


        # === 5. Assemble Output ===
        return {
            "tokens": tokens,
            "masks": {
                "action": self.get_action_mask(agent),
                "attention": attention_mask
            }
        }



    


    # def build_transformer_state(self, agent, phase="active", **kwargs):
    #     # === Basic info ===
    #     state_vec = torch.tensor([
    #         agent.tier,
    #         agent.health,
    #         self.turn
    #     ], dtype=torch.float32)

    #     # === Econ info ===
    #     if phase == "active":
    #         econ_vec = torch.tensor([
    #             agent.gold,
    #             agent.gold_cap,
    #             1.0,               # fixed roll cost
    #             agent.tavern_upgrade_cost
    #         ], dtype=torch.float32)

    #         shop_minions = []
    #         for i in range(self.MAX_SHOP_SIZE):  # or range(6) if you prefer
    #             if i < len(agent.shop):
    #                 shop_minions.append(self.encode_minion(agent.shop[i], source_flag=0, slot_idx=i))
    #             else:
    #                 shop_minions.append(self.encode_minion(None, source_flag=0, slot_idx=i))  # empty slot


    #         opponent = self.current_opponent.get(agent, None)
    #         opponent_vec = None
    #         if opponent in agent.opponent_memory:
    #             opponent_vec = torch.tensor(agent.opponent_memory[opponent], dtype=torch.float32)

    #     if phase == "combat":
    #         # === Combat econ vector (no shop)
    #         econ_vec = torch.tensor([0.0, agent.gold_cap, 0.0, 0.0], dtype=torch.float32)

    #         # === No shop minions during combat
    #         shop_minions = [
    #             self.encode_minion(None, source_flag=0, slot_idx=i)
    #             for i in range(self.MAX_SHOP_SIZE)  # or 6, depending on your standard
    #         ]

    #         # === Opponent vector using passed-in kwargs
    #         hp_delta = kwargs.get("hp_delta", 0.0)
    #         opponent_strength = kwargs.get("opponent_strength", 0.0)

    #         opponent = self.current_opponent.get(agent, None)
    #         opponent_vec = None

    #         if opponent is not None:
    #             summary = [
    #                 float(opponent.tier),
    #                 float(opponent_strength),
    #                 float(hp_delta),
    #                 self.turn
    #             ]
    #             agent.opponent_memory[opponent] = summary
    #             opponent_vec = torch.tensor(summary, dtype=torch.float32)

                

    #     # === Tier vector ===
    #     tier_vec = torch.tensor([
    #         1 if i < agent.tier else 0 for i in range(self.MAX_TIER)
    #     ], dtype=torch.float32)

    #     board_minions = []
    #     for i in range(7):  # Always 7 board slots
    #         if i < len(agent.board):
    #             board_minions.append(self.encode_minion(agent.board[i], source_flag=1, slot_idx=i))
    #         else:
    #             board_minions.append(self.encode_minion(None, source_flag=1, slot_idx=i))  # padded empty slot


    #     return agent.build_tokens(
    #         state_vec=state_vec,
    #         board_minions=board_minions,
    #         shop_minions=shop_minions,
    #         econ_vec=econ_vec,
    #         tier_vec=tier_vec,
    #         current_turn=self.turn,  # Explicitly passed here
    #         opponent_vec=opponent_vec
    #     )

    def build_state(self, agent, phase="active", **kwargs):
        """Polymorphic state builder that works for both agent types"""
        if isinstance(agent, TransformerAgent):
            # Transformer gets full tokenized state
            return self.build_transformer_state(agent, phase, **kwargs)
        else:
            # MLP gets simplified vector
            return self.build_mlp_state(agent, phase, **kwargs)

    def build_mlp_state(self, agent, phase="active", **kwargs):
        turn_number = self.turn
        tier = agent.tier
        gold_cap = agent.gold_cap
        num_minions = len(agent.board)
        board_strength = sum(m.strength() for m in agent.board)

        if phase == "active":
            gold = agent.gold
            health = agent.health
            shop_slots = [m.tier for m in agent.shop]
            while len(shop_slots) < self.MAX_SHOP_SIZE:
                shop_slots.append(0)
            health_delta = 0.0
        elif phase == "combat":
            gold = 0.0
            health = agent.health
            shop_slots = [0.0] * self.MAX_SHOP_SIZE
            health_delta = kwargs.get("hp_delta", 0.0)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        state_vector = [
            gold, gold_cap, tier, health, turn_number,
            board_strength, num_minions,
            *shop_slots,
            health_delta,
            kwargs.get("opponent_strength", 0.0)
        ]
        while len(state_vector) < 19:
            state_vector.append(0.0)
        return torch.tensor(state_vector, dtype=torch.float32)



import torch
from types import SimpleNamespace

def test_encode_minion():
    encode = TinyBattlegroundsEnv.encode_minion  # Shortcut if inside TransformerAgent
    
    # Case 1: Normal minion with "Beast"
    m1 = SimpleNamespace(attack=3, health=2, tier=1, types=["Beast"])
    out1 = encode(None, 0)
    print(out1)

# test_encode_minion()
